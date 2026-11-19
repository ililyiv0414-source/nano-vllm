"""Replay a benchmark record with CPU/CUDA ranges. Diagnostic, not a benchmark."""
import argparse
from contextlib import ExitStack
import functools
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--result', required=True, help='One benchmark_text worker JSON')
    parser.add_argument('--out', required=True)
    parser.add_argument('--greedy-fastpath', action='store_true')
    a = parser.parse_args()
    record = json.loads(Path(a.result).read_text(encoding='utf-8'))
    folder = Path(a.out)
    folder.mkdir(parents=True, exist_ok=False)
    import torch
    from nanovllm import LLM, SamplingParams
    from experiments.prefix_cache import source_digest

    torch.manual_seed(42)
    w = record['workload']
    llm = LLM(a.model, enforce_eager=True, enable_prefix_caching=bool(record['cache']),
              enable_greedy_fastpath=a.greedy_fastpath, max_num_seqs=record['batch'],
              max_model_len=w['input_tokens'] + w['output_tokens'],
              max_num_batched_tokens=w['input_tokens'] * record['batch'], gpu_memory_utilization=0.8)
    try:
        sp = SamplingParams(temperature=0, max_tokens=w['output_tokens'], ignore_eos=True)
        for phase in ('cold', 'warm'):
            llm.generate(record[phase]['prompts'], sp, use_tqdm=False)
        bm = llm.scheduler.block_manager
        assert not bm.used_block_ids
        bm.hash_to_block_id.clear()
        for block in bm.blocks:
            block.hash = -1
            block.token_ids = []
        torch.cuda.synchronize()

        def wrap(fn, label):
            @functools.wraps(fn)
            def measured(*args, **kwargs):
                with torch.profiler.record_function(label):
                    return fn(*args, **kwargs)
            return measured

        runner = llm.model_runner
        targets = [(llm.scheduler, 'schedule'), (llm.scheduler, 'postprocess'),
                   (runner, 'prepare_prefill'), (runner, 'prepare_decode'),
                   (runner, 'prepare_sample'), (runner, 'run_model'), (runner, 'run'),
                   (runner.sampler, 'forward'), (runner.sampler, 'greedy'),
                   (llm.tokenizer, 'decode')]
        measurements = {}
        with ExitStack() as stack:
            for obj, name in targets:
                label = f'phase::{type(obj).__name__}.{name}'
                stack.enter_context(patch.object(obj, name, wrap(getattr(obj, name), label)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                for phase in ('cold', 'warm'):
                    with torch.profiler.record_function(f'generation::{phase}'):
                        outputs = llm.generate(record[phase]['prompts'], sp, use_tqdm=False)
                        torch.cuda.synchronize()
                    ids = [o['token_ids'] for o in outputs]
                    assert ids == record[phase]['outputs'], f'{phase}: output mismatch'
                    measurements[phase] = dict(output_equal=True, stats=dict(llm.last_batch_stats))
        prof.export_chrome_trace(str(folder / 'trace.json'))
        averages = prof.key_averages()
        for sort, filename in [('self_cpu_time_total', 'cpu.txt'), ('self_cuda_time_total', 'cuda.txt')]:
            (folder / filename).write_text(averages.table(sort_by=sort, row_limit=40), encoding='utf-8')
        ranges = []
        for event in averages:
            if event.key.startswith(('phase::', 'generation::')) and event.cpu_time_total > 0:
                ranges.append(dict(name=event.key, calls=event.count,
                                   cpu_total_us=event.cpu_time_total, cpu_self_us=event.self_cpu_time_total,
                                   device_total_us=getattr(event, 'device_time_total', None)))
        result = dict(diagnostic_only=True, source=source_digest(), input_result=str(Path(a.result).resolve()),
                      greedy_fastpath=a.greedy_fastpath, cache=record['cache'], workload=w,
                      torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0),
                      measurements=measurements, ranges=ranges,
                      caveat='Nested inclusive ranges must not be summed. CPU ranges include waits; GPU launches are asynchronous. Profile timings are not benchmark results.')
        (folder / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
