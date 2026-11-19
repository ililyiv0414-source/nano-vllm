"""Separate reference precision from full-recompute/incremental execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=('all', 'bf16-full', 'bf16-cache', 'fp32-full'), default='all')
    args = parser.parse_args()
    out = Path(args.out)
    case = Path(args.case_dir)
    if args.mode == 'all':
        if out.exists():
            parser.error('Use a new output directory')
        assert (case / 'teacher/off.pt').exists()
        out.mkdir(parents=True)
        for mode in ('bf16-full', 'bf16-cache', 'fp32-full'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', args.model,
                            '--case-dir', str(case), '--out', str(out), '--mode', mode], check=True)
        import torch
        data = {name: torch.load(out / f'{name}.pt', weights_only=True, map_location='cpu')
                for name in ('bf16-full', 'bf16-cache', 'fp32-full')}
        data['nano-off'] = torch.load(case / 'teacher/off.pt', weights_only=True, map_location='cpu')
        data['previous-hf'] = torch.load(case / 'teacher/hf.pt', weights_only=True, map_location='cpu')
        anchor = data['fp32-full']
        for d in data.values():
            assert d['prompts'] == anchor['prompts'] and d['teacher'] == anchor['teacher']
            assert d['logits'].shape == anchor['logits'].shape
        rows = []
        pairs = [('previous-hf', 'bf16-full'), ('nano-off', 'bf16-full'),
                 ('nano-off', 'bf16-cache'), ('bf16-cache', 'bf16-full'),
                 ('nano-off', 'fp32-full'), ('bf16-full', 'fp32-full'), ('bf16-cache', 'fp32-full')]
        for left, right in pairs:
            x, y = data[left]['logits'], data[right]['logits']
            for step in range(x.shape[0]):
                for request in range(x.shape[1]):
                    u, v = x[step, request], y[step, request]
                    delta = u - v
                    centered = delta - delta.mean()
                    row = dict(pair=f'{left}/{right}', step=step, request=request,
                               context_length=len(anchor['prompts'][request])+step,
                               finite=bool(torch.isfinite(u).all() and torch.isfinite(v).all()),
                               max_abs=delta.abs().max().item(), rms=delta.square().mean().sqrt().item(),
                               centered_rms=centered.square().mean().sqrt().item(),
                               left_argmax=u.argmax().item(), right_argmax=v.argmax().item())
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        (out / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print('Diagnostic only: FP32 is a stronger reference, not exact arithmetic or automatic acceptance.')
        return

    import torch
    from transformers import AutoModelForCausalLM
    raw = json.loads((case / 'baseline.json').read_text(encoding='utf-8'))
    warm = next(r for r in raw['records'] if r['phase'] == 'warm')
    prompts, teacher = warm['prompts'], warm['outputs']
    steps = raw['parameters']['output_length']
    assert all(len(t) == steps for t in teacher)
    target = out / f'{args.mode}.pt'
    assert not target.exists()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if args.mode == 'fp32-full' else torch.bfloat16
    cached = args.mode == 'bf16-cache'
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                torch_dtype=dtype, attn_implementation='eager').cuda().eval()
    captured, past = [], None
    with torch.inference_mode():
        for step in range(steps):
            tokens = ([[t[step-1]] for t in teacher] if cached and step else
                      [prompt + t[:step] for prompt, t in zip(prompts, teacher)])
            ids = torch.tensor(tokens, device='cuda')
            result = model(input_ids=ids, past_key_values=past, use_cache=cached)
            captured.append(result.logits[:, -1].float().cpu())
            if cached:
                past = result.past_key_values
                assert past.get_seq_length() == len(prompts[0]) + step
    torch.save(dict(prompts=prompts, teacher=teacher, logits=torch.stack(captured),
                    mode=args.mode, torch_version=str(torch.__version__)), target)
    print('Saved', target, flush=True)


if __name__ == '__main__':
    main()
