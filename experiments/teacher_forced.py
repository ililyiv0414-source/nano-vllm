"""Fixed-history diagnostics, never a generation benchmark or token-equality pass."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--result', required=True, help='cache-off original worker JSON')
    p.add_argument('--out', required=True, help='New directory')
    p.add_argument('--backend', choices=('all', 'off', 'on', 'hf'), default='all')
    a = p.parse_args()
    out = Path(a.out)
    if a.backend == 'all':
        if out.exists():
            p.error('Use a new output directory')
        out.mkdir(parents=True)
        for backend in ('off', 'on', 'hf'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', a.model,
                            '--result', a.result, '--out', str(out), '--backend', backend], check=True)
        import torch
        data = {b: torch.load(out / f'{b}.pt', weights_only=True, map_location='cpu') for b in ('off', 'on', 'hf')}
        assert data['off']['prompts'] == data['on']['prompts'] == data['hf']['prompts']
        assert data['off']['teacher'] == data['on']['teacher'] == data['hf']['teacher']
        rows = []
        for left, right in (('off', 'on'), ('off', 'hf'), ('on', 'hf')):
            x, y = data[left]['logits'], data[right]['logits']
            assert x.shape == y.shape
            for step in range(x.shape[0]):
                for request in range(x.shape[1]):
                    u, v = x[step, request], y[step, request]
                    delta = u - v
                    uid, vid = u.argmax().item(), v.argmax().item()
                    ut, vt = u.topk(2).values, v.topk(2).values
                    row = dict(pair=f'{left}/{right}', step=step, request=request,
                               finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all()),
                               max_abs=delta.abs().max().item(), rms=delta.square().mean().sqrt().item(),
                               left_argmax=uid, right_argmax=vid, same_argmax=uid == vid,
                               left_margin=(ut[0]-ut[1]).item(), right_margin=(vt[0]-vt[1]).item())
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        (out / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print('Diagnostic complete; no numerical tolerance or benchmark acceptance inferred.')
        return

    import torch
    source = json.loads(Path(a.result).read_text(encoding='utf-8'))
    params = source['parameters']
    assert params['enabled'] == 0, 'Teacher must come from cache-off baseline'
    warm = next(r for r in source['records'] if r['phase'] == 'warm')
    cold = next(r for r in source['records'] if r['phase'] == 'cold')
    prompts, teacher = warm['prompts'], warm['outputs']
    batch, steps = params['batch_size'], params['output_length']
    assert len(prompts) == len(teacher) == batch
    assert all(len(t) == steps for t in teacher)
    assert all(len(t) == params['input_length'] for t in prompts)
    torch.manual_seed(params['seed'] + params['repeat'])
    captured = []
    target = out / f'{a.backend}.pt'
    assert not target.exists()
    if a.backend == 'hf':
        from transformers import AutoModelForCausalLM
        # Independent eager reference, same BF16 weights; not an FP64 model oracle.
        model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
        with torch.inference_mode():
            for step in range(steps):
                ids = torch.tensor([prompt + tokens[:step] for prompt, tokens in zip(prompts, teacher)], device='cuda')
                logits = model(input_ids=ids, use_cache=False).logits[:, -1].float().cpu()
                captured.append(logits)
    else:
        from nanovllm import LLM, SamplingParams
        llm = LLM(a.model, enforce_eager=True, enable_prefix_caching=a.backend == 'on',
                  max_model_len=params['input_length'] + steps, max_num_seqs=batch,
                  max_num_batched_tokens=batch * params['input_length'], gpu_memory_utilization=0.8)
        try:
            sp = SamplingParams(temperature=0, max_tokens=steps, ignore_eos=True)
            llm.generate([[1, 2, 3, 4]] * batch, sp, use_tqdm=False)
            bm = llm.scheduler.block_manager
            assert not bm.used_block_ids
            bm.hash_to_block_id.clear()
            for block in bm.blocks:
                block.hash = -1
                block.token_ids = []
            llm.generate(cold['prompts'], sp, use_tqdm=False)
            runner = llm.model_runner
            original_model, original_run = runner.run_model, runner.run
            def capture(*args, **kwargs):
                logits = original_model(*args, **kwargs)
                captured.append(logits.detach().float().cpu().clone())
                return logits
            def force(seqs, is_prefill):
                step = len(captured)
                assert step < steps and len(seqs) == batch
                for row, seq in enumerate(seqs):
                    assert seq.token_ids == prompts[row] + teacher[row][:step], 'History/order mismatch'
                original_run(seqs, is_prefill)
                assert len(captured) == step + 1
                return [tokens[step] for tokens in teacher]
            runner.run_model, runner.run = capture, force
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            assert [o['token_ids'] for o in outputs] == teacher
            # This equality verifies forced history only, not model correctness.
            print(a.backend, 'forced histories verified; stats:', llm.last_batch_stats, flush=True)
        finally:
            llm.exit()
    assert len(captured) == steps
    torch.save(dict(prompts=prompts, teacher=teacher, logits=torch.stack(captured), backend=a.backend,
                    torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0)), target)
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
