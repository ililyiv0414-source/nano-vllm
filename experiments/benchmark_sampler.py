"""Compiled CUDA sampler A/B microbenchmark; not end-to-end throughput."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    if a.iterations < 1 or a.repeats < 1:
        p.error('Counts must be positive')
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    import torch
    from nanovllm.layers.sampler import Sampler
    vocab = json.loads(Path(a.model, 'config.json').read_text())['vocab_size']
    torch.manual_seed(42)
    sampler = Sampler().cuda()
    rows = []
    for batch in (1, 4, 16):
        logits = torch.randn(batch, vocab, device='cuda', dtype=torch.bfloat16)
        temperatures = torch.zeros(batch, device='cuda')
        paths = {'baseline': lambda: sampler(logits, temperatures),
                 'fast': lambda: sampler.greedy(logits)}
        for fn in paths.values():
            for _ in range(10):
                fn()
        torch.cuda.synchronize()
        assert torch.equal(paths['baseline'](), paths['fast']())
        assert torch.equal(paths['fast'](), logits.argmax(-1))
        times = {name: [] for name in paths}
        for repeat in range(a.repeats):
            for name in (('baseline', 'fast') if repeat % 2 == 0 else ('fast', 'baseline')):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(a.iterations):
                    paths[name]()
                end.record()
                end.synchronize()
                times[name].append(start.elapsed_time(end) / a.iterations)
        row = dict(batch=batch, milliseconds=times,
                   median_paired_ratio=statistics.median(x/y for x,y in zip(times['baseline'], times['fast'])))
        rows.append(row)
        print(json.dumps(row), flush=True)
        # Profiling is separate from timing and runs only after compile/warmup.
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for name, fn in paths.items():
                    with torch.profiler.record_function(name):
                        for _ in range(5):
                            fn()
                        torch.cuda.synchronize()
            prof.export_chrome_trace(str(folder / f'b{batch}-trace.json'))
            (folder / f'b{batch}-profile.txt').write_text(
                prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30), encoding='utf-8')
    result = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), dtype='bfloat16', vocab=vocab,
                  iterations=a.iterations, repeats=a.repeats, rows=rows,
                  scope='Compiled sampler only, preallocated temperatures; excludes model and CPU parameter preparation')
    (folder / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
