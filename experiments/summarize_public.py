"""Validate sanitized evidence and recompute paired medians, without GPU dependencies."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def summarize(data):
    rows = data['records']
    indexed = {(r['batch'], r['repeat'], r['cache'], r['phase'], r['mode']): r for r in rows}
    expected = {(b,r,c,p,m) for b in (1,4,16) for r in range(5)
                for c in (0,1) for p in ('cold','warm') for m in ('base','fast')}
    if len(rows) != len(indexed) or set(indexed) != expected:
        raise ValueError('Duplicate or incomplete evidence matrix')
    summary = []
    for b in (1,4,16):
        for c in (0,1):
            for p in ('cold','warm'):
                ratios=[]
                for r in range(5):
                    a,z=[indexed[b,r,c,p,m] for m in ('base','fast')]
                    for key in ('prompts_sha256','outputs_sha256','prompts_lengths','outputs_lengths','stats','source_sha256'):
                        if a[key] != z[key]:
                            raise ValueError(f'Pair mismatch: {b}/{r}/{c}/{p}/{key}')
                    for row in (a,z):
                        if row['prompts_lengths'] != [512]*b or row['outputs_lengths'] != [64]*b:
                            raise ValueError('Wrong workload lengths')
                        if not math.isfinite(row['seconds']) or row['seconds'] <= 0:
                            raise ValueError('Invalid time')
                        if row['stats']['preemptions'] or row['stats']['used_blocks']:
                            raise ValueError('Unexpected preemption or active blocks')
                    ratios.append(a['seconds']/z['seconds'])
                summary.append(dict(batch=b, cache=c, phase=p, ratios=ratios,
                                    speedup=statistics.median(ratios),
                                    reduction=100*statistics.median(1-1/v for v in ratios)))
    return summary


if __name__ == '__main__':
    data = json.loads((ROOT/'benchmarks/a800-greedy.json').read_text(encoding='utf-8'))
    print('| Batch | Prefix cache | Phase | Paired speedup | Latency reduction |')
    print('|---|---|---|---|---|')
    for r in summarize(data):
        print(f"| {r['batch']} | {r['cache']} | {r['phase']} | {r['speedup']:.4f}x | {r['reduction']:.2f}% |")
