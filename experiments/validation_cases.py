"""Text-derived controlled workloads; diagnostic results are not performance claims."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = {
    'boundary255': (255, 128, 2),
    'boundary256': (256, 128, 2),
    'boundary257': (257, 256, 2),
    'prefix512': (512, 256, 2),
    'batch4': (512, 256, 4),
}


def build_inputs(tokenizer, length, prefix, batch):
    def exact(text, n):
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids
        return (ids * ((n + len(ids) - 1) // len(ids)))[:n]
    shared = exact('这是一个关于推理系统的技术说明。模型逐步生成文本，缓存保存历史键和值。请根据上下文继续回答。\n', prefix)
    # A distinct ordinary token starts each suffix, guaranteeing the exact shared-prefix boundary.
    markers = []
    for word in ('苹果', '香蕉', '城市', '河流', '数学', '历史', '音乐', '运动', '天气', '植物'):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids and ids[0] not in markers and ids[0] not in tokenizer.all_special_ids:
            markers.append(ids[0])
    assert len(markers) >= batch * 2
    suffix = exact('请解释缓存如何减少重复计算，并给出一个简短例子。\n', length - prefix - 1)
    return [[shared + [markers[phase * batch + row]] + suffix for row in range(batch)] for phase in range(2)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--baseline-only', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    folder = Path(a.out)
    if not a.baseline_only:
        if folder.exists():
            p.error('Use a new directory; previous evidence is preserved')
        folder.mkdir(parents=True)
        # Keep CUDA out of the coordinator: deleting the runner does not ensure
        # its allocator reservations/context are returned to another process.
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', a.model,
                        '--case', a.case, '--out', str(folder), '--baseline-only'], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('teacher_forced.py')),
                        '--model', a.model, '--result', str(folder / 'baseline.json'),
                        '--out', str(folder / 'teacher')], check=True)
        return
    assert folder.is_dir() and not (folder / 'baseline.json').exists()
    length, prefix, batch = CASES[a.case]
    import torch
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(42)
    llm = LLM(a.model, enforce_eager=True, enable_prefix_caching=False,
              max_model_len=length + 8, max_num_seqs=batch,
              max_num_batched_tokens=batch * length, gpu_memory_utilization=0.8)
    try:
        cold, warm = build_inputs(llm.tokenizer, length, prefix, batch)
        sp = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        llm.generate([[1, 2, 3, 4]] * batch, sp, use_tqdm=False)
        records = []
        for phase, prompts in [('cold', cold), ('warm', warm)]:
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            records.append(dict(phase=phase, prompts=prompts,
                                outputs=[o['token_ids'] for o in outputs]))
        data = dict(parameters=dict(seed=42, repeat=0, enabled=0, input_length=length,
                                   output_length=8, batch_size=batch, prefix=prefix), records=records,
                    case=a.case, note='Repeated/truncated text tokens plus distinct suffix markers; not natural chat quality evaluation',
                    preview=llm.tokenizer.decode(warm[0]))
        (folder / 'baseline.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
