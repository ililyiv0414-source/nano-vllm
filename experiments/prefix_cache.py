"""Run each cache/prefix/repeat in a fresh process; GPU required only by workers."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serializable_model_config(config):
    """Normalize precision metadata without mutating the runtime config."""
    import torch

    result = dict(config.to_dict())
    for key in ('dtype', 'torch_dtype'):
        if isinstance(result.get(key), torch.dtype):
            result[key] = str(result[key])
    return result


def make_batches(tokenizer, batch_size, input_length, prefix_length, seed):
    """Token-ID workload: exact prefix lengths, distinct first suffix tokens."""
    rng = random.Random(seed)
    excluded = set(tokenizer.all_special_ids)
    pool = [i for i in range(tokenizer.vocab_size) if i not in excluded]
    markers = rng.sample(pool, batch_size * 2 + 1)
    shared = [markers[-1]] + rng.choices(pool, k=max(0, prefix_length - 1)) if prefix_length else []
    batches = []
    for phase in range(2):
        batches.append([shared + [markers[phase * batch_size + i]] +
                        rng.choices(pool, k=input_length - prefix_length - 1)
                        for i in range(batch_size)])
    return batches


def source_digest():
    h = hashlib.sha256()
    for folder in (ROOT / "nanovllm", ROOT / "experiments"):
        for path in sorted(folder.rglob("*.py")):
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def command_output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def model_digest(model):
    h = hashlib.sha256()
    files = sorted(Path(model).glob('*.safetensors'))
    if not files:
        raise ValueError('Expected safetensors model weights')
    for path in files:
        h.update(path.name.encode())
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def worker(args):
    import torch
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(args.seed + args.repeat)
    llm = LLM(args.model, enforce_eager=True, enable_prefix_caching=bool(args.enabled),
              max_model_len=args.input_length + args.output_length,
              max_num_seqs=args.batch_size, max_num_batched_tokens=args.batch_size * args.input_length,
              gpu_memory_utilization=0.8)
    try:
        batches = make_batches(llm.tokenizer, args.batch_size, args.input_length,
                               args.prefix, args.seed + args.repeat)
        sp = SamplingParams(temperature=0, max_tokens=args.output_length, ignore_eos=True)
        # Warm kernels with unrelated, shorter inputs, then clear reusable metadata.
        llm.generate([[1, 2, 3, 4]] * args.batch_size, sp, use_tqdm=False)
        bm = llm.scheduler.block_manager
        assert not bm.used_block_ids
        bm.hash_to_block_id.clear()
        for block in bm.blocks:
            block.hash = -1
            block.token_ids = []
        records = []
        for phase, prompts in zip(("cold", "warm"), batches):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats = dict(llm.last_batch_stats)
            stats.update(batch_seconds=elapsed,
                         output_tokens_per_second=stats["output_tokens"] / elapsed,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated())
            assert stats["used_blocks"] == 0, "request-owned blocks leaked"
            assert stats["input_tokens"] == args.batch_size * args.input_length
            assert stats["input_tokens"] == stats["reused_tokens"] + stats["pending_input_tokens"]
            assert not stats["preemptions"], "reduce batch size uniformly; preemption confounds experiment"
            if not args.enabled or args.prefix < bm.block_size:
                assert stats["reused_tokens"] == 0
            if args.enabled and phase == "warm" and args.prefix >= bm.block_size:
                assert stats["reused_tokens"] > 0, "expected prefix was evicted or not reusable"
            records.append(dict(phase=phase, stats=stats, prompts=prompts,
                                outputs=[o["token_ids"] for o in outputs]))
        result = dict(parameters=vars(args), source_sha256=source_digest(),
                      upstream=json.loads((ROOT / "upstream-commit.json").read_text())["sha"],
                      local_git_head=command_output(["git", "rev-parse", "HEAD"]),
                      environment=dict(python=platform.python_version(), torch=torch.__version__,
                                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                                       dtype=str(llm.model_runner.config.hf_config.dtype),
                                       model_config=serializable_model_config(llm.model_runner.config.hf_config),
                                       model_config_sha256=hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
                                       model_weights_sha256=model_digest(args.model),
                                       pip_freeze=command_output([sys.executable, "-m", "pip", "freeze"]),
                                       nvidia_smi=command_output(["nvidia-smi"])), records=records)
        Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        llm.exit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="local model snapshot directory")
    p.add_argument("--out", default="results/prefix-cache")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--input-length", type=int, default=512)
    p.add_argument("--output-length", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hourly-rate", type=float, default=3., help="actual instance CNY/hour")
    p.add_argument("--budget", type=float, default=300.)
    p.add_argument("--already-spent", type=float, default=0., help="include prior rental and storage charges")
    p.add_argument("--reserve", type=float, default=30., help="keep this amount unspent")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--enabled", type=int, choices=(0, 1), default=1, help=argparse.SUPPRESS)
    p.add_argument("--prefix", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--repeat", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.batch_size, args.output_length, args.repeats) <= 0 or args.input_length <= 256:
        p.error("positive counts and input-length > 256 required")
    if args.hourly_rate <= 0 or min(args.already_spent, args.reserve) < 0:
        p.error("invalid cost settings")
    if args.worker:
        worker(args)
        return
    remaining = args.budget - args.already_spent - args.reserve
    if remaining <= 0:
        p.error("budget reserve reached: no experiments started")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error("output directory must be empty to avoid mixing runs")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + remaining / args.hourly_rate * 3600
    print("Budget guard covers this script's wall time only. Shut down the cloud instance yourself.", flush=True)
    try:
        for repeat in range(args.repeats):
            for prefix in (0, 128, 256):
                # Alternate execution order to reduce systematic warm-machine bias.
                for enabled in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    target = out / f"r{repeat}-p{prefix}-cache{enabled}.json"
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--model", str(Path(args.model).resolve()), "--batch-size", str(args.batch_size),
                           "--input-length", str(args.input_length), "--output-length", str(args.output_length),
                           "--seed", str(args.seed), "--repeat", str(repeat), "--prefix", str(prefix),
                           "--repeats", str(args.repeats),
                           "--enabled", str(enabled), "--result", str(target)]
                    seconds_left = deadline - time.monotonic()
                    if seconds_left <= 0:
                        raise RuntimeError("budget reserve reached")
                    print(f"Running {target.name}", flush=True)
                    subprocess.run(cmd, check=True, timeout=seconds_left, cwd=ROOT)
        subprocess.run([sys.executable, str(ROOT / "experiments/report.py"), str(out)], check=True)
    finally:
        (out / "cost-estimate.json").write_text(json.dumps(dict(
            hourly_rate=args.hourly_rate, already_spent=args.already_spent,
            session_estimate=(time.monotonic() - started) / 3600 * args.hourly_rate,
            note="Not a provider invoice; instance keeps billing after this script exits."), indent=2))


if __name__ == "__main__":
    main()
