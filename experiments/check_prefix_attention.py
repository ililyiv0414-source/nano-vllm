"""Diagnostic only: compare paged suffix attention with a dense FP32 oracle."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/attention-check-01.json')
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        parser.error('Output already exists; choose a new path')
    import torch
    from flash_attn import flash_attn_varlen_func

    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with torch.inference_mode():
        for seed in (42, 43, 44):
            for dtype in (torch.bfloat16, torch.float16):
                torch.manual_seed(seed)
                # Qwen3-0.6B head geometry; two requests sharing 256 prefix tokens.
                batch, length, prefix, heads, kvheads, dim = 2, 512, 256, 16, 8, 128
                q = torch.randn(batch, length, heads, dim, device='cuda', dtype=dtype)
                k = torch.randn(batch, length, kvheads, dim, device='cuda', dtype=dtype)
                v = torch.randn_like(k)
                k[1, :prefix].copy_(k[0, :prefix])
                v[1, :prefix].copy_(v[0, :prefix])
                cu = torch.arange(0, (batch + 1) * length, length, device='cuda', dtype=torch.int32)
                full = flash_attn_varlen_func(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
                                             cu, cu, length, length, causal=True).reshape_as(q)
                suffix = q[:, prefix:].reshape(-1, heads, dim).contiguous()
                cuq = torch.arange(0, (batch + 1) * prefix, prefix, device='cuda', dtype=torch.int32)
                # Explicit bottom-right causal alignment: suffix query i is at position 256+i.
                keys = k.float().repeat_interleave(heads // kvheads, dim=2).transpose(1, 2)
                values = v.float().repeat_interleave(heads // kvheads, dim=2).transpose(1, 2)
                queries = q[:, prefix:].float().transpose(1, 2)
                scores = queries @ keys.transpose(-1, -2) / dim ** .5
                mask = torch.arange(length, device='cuda')[None, :] <= torch.arange(prefix, length, device='cuda')[:, None]
                reference = (scores.masked_fill(~mask, -float('inf')).softmax(-1) @ values).transpose(1, 2)
                layouts = {}
                for layout, table in [('ordered', [[0, 1], [0, 2]]), ('permuted', [[3, 0], [3, 2]])]:
                    kc = torch.zeros(4, prefix, kvheads, dim, device='cuda', dtype=dtype)
                    vc = torch.zeros_like(kc)
                    for row in range(batch):
                        for block in range(2):
                            physical = table[row][block]
                            kc[physical].copy_(k[row, block * prefix:(block + 1) * prefix])
                            vc[physical].copy_(v[row, block * prefix:(block + 1) * prefix])
                    output = flash_attn_varlen_func(suffix, kc, vc, cuq, cu, prefix, length,
                                                   causal=True, block_table=torch.tensor(table, device='cuda', dtype=torch.int32))
                    layouts[layout] = output.reshape(batch, prefix, heads, dim).float()
                baseline = full[:, prefix:].float()
                def error(x, y):
                    d = (x - y).abs()
                    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(), rms=d.square().mean().sqrt().item())
                result = dict(seed=seed, dtype=str(dtype),
                              finite=all(torch.isfinite(x).all().item() for x in [reference, baseline, *layouts.values()]),
                              full_vs_fp32=error(baseline, reference),
                              paged_vs_fp32=error(layouts['ordered'], reference),
                              paged_vs_full=error(layouts['ordered'], baseline),
                              physical_permutation=error(layouts['ordered'], layouts['permuted']))
                records.append(result)
                print(json.dumps(result), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                                   note='Isolated random QKV diagnostic; not full-model correctness or performance acceptance',
                                   records=records), indent=2), encoding='utf-8')
    print('Saved:', path)


if __name__ == '__main__':
    main()
