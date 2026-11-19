"""Offline text-derived batch benchmark. No logits capture or online-latency claims."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.prefix_cache import source_digest, model_digest, command_output


def summarize(folder):
    records = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('b*-r*-c*.json'))]
    plan = json.loads((folder / 'plan.json').read_text())
    expected = {(b,r,c) for b in plan['batches'] for r in range(plan['repeats']) for c in (0,1)}
    indexed = {(d['batch'],d['repeat'],d['cache']): d for d in records}
    assert set(indexed) == expected and len(records) == len(expected), 'Incomplete matrix'
    assert len({json.dumps(d['environment'], sort_keys=True) for d in records}) == 1, 'Environment changed'
    lines = ['# Text-derived batch benchmark', '', 'Seconds include generate and GPU synchronization; not TTFT. No diagnostic tensor capture.', '',
             '| Batch | Phase | Token equality | Off median s | On median s | Median paired off/on | Off median output tok/s | On median output tok/s |',
             '|---|---|---|---|---|---|---|---|']
    for batch in plan['batches']:
        for phase in ('cold','warm'):
            times = {0:[],1:[]}; rates = {0:[],1:[]}; paired=[]; same=True
            for repeat in range(plan['repeats']):
                a,b = [indexed[batch,repeat,c][phase] for c in (0,1)]
                assert a['prompts'] == b['prompts']
                same = same and a['outputs'] == b['outputs']
                for c,row in enumerate((a,b)):
                    times[c].append(row['seconds']); rates[c].append(row['output_tok_s'])
                paired.append(a['seconds']/b['seconds'])
            ratio = f'{statistics.median(paired):.3f}x' if same else 'WITHHELD: output mismatch'
            lines.append(f'| {batch} | {phase} | {same} | {statistics.median(times[0]):.6f} | {statistics.median(times[1]):.6f} | {ratio} | {statistics.median(rates[0]):.2f} | {statistics.median(rates[1]):.2f} |')
    lines += ['', 'Raw repeats, output IDs, cache work counters and PyTorch memory peaks are retained in worker JSON.',
              'KV pool is preallocated: peak allocated bytes is not effective cache occupancy. Text is repeated/truncated for length control; this is not a quality benchmark.']
    (folder/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[1,4,16])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--input-length', type=int, default=512)
    p.add_argument('--prefix-length', type=int, default=256)
    p.add_argument('--output-length', type=int, default=64)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--greedy-fastpath', action='store_true')
    p.add_argument('--cuda-graph', action='store_true', help='Use existing decode CUDA Graph path instead of eager')
    p.add_argument('--batch', type=int)
    p.add_argument('--repeat', type=int)
    p.add_argument('--cache', type=int, choices=(0,1))
    a=p.parse_args(); folder=Path(a.out)
    if not (0 < a.prefix_length < a.input_length and a.prefix_length % 256 == 0 and a.output_length > 0):
        p.error('Require 0 < prefix < input, prefix divisible by 256, positive output')
    if not a.worker:
        if folder.exists() or a.repeats<1 or not a.batches or min(a.batches)<1 or len(set(a.batches))!=len(a.batches):
            p.error('Use a new directory, positive counts and distinct batches')
        folder.mkdir(parents=True)
        (folder/'plan.json').write_text(json.dumps(dict(batches=a.batches,repeats=a.repeats,input_tokens=a.input_length,prefix_tokens=a.prefix_length,output_tokens=a.output_length)),encoding='utf-8')
        for batch in a.batches:
            for repeat in range(a.repeats):
                for cache in ((0,1) if repeat%2==0 else (1,0)):
                    print('Running',batch,repeat,cache,flush=True)
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--model',a.model,
                                    '--out',str(folder),'--batch',str(batch),'--repeat',str(repeat),'--cache',str(cache),
                                    '--input-length',str(a.input_length),'--prefix-length',str(a.prefix_length),
                                    '--output-length',str(a.output_length)] + (['--greedy-fastpath'] if a.greedy_fastpath else []) + (['--cuda-graph'] if a.cuda_graph else []),check=True)
        summarize(folder); return
    import torch
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(42)
    target=folder/f'b{a.batch}-r{a.repeat}-c{a.cache}.json'
    assert not target.exists()
    environment=dict(source=source_digest(),weights=model_digest(a.model),
                     torch=str(torch.__version__),cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0),pip=command_output([sys.executable,'-m','pip','freeze']))
    environment['greedy_fastpath']=a.greedy_fastpath
    environment['enforce_eager']=not a.cuda_graph
    llm=LLM(a.model,enforce_eager=not a.cuda_graph,enable_prefix_caching=bool(a.cache),enable_greedy_fastpath=a.greedy_fastpath,
            max_model_len=a.input_length+a.output_length,max_num_seqs=a.batch,max_num_batched_tokens=a.input_length*a.batch,gpu_memory_utilization=0.8)
    try:
        environment['dtype']=str(llm.model_runner.config.hf_config.dtype)
        environment['config']=Path(a.model,'config.json').read_text()
        environment['gpu_uuid']=str(torch.cuda.get_device_properties(0).uuid)
        def exact(text,n):
            ids=llm.tokenizer.encode(text,add_special_tokens=False)
            return (ids*((n+len(ids)-1)//len(ids)))[:n]
        prefix=exact('实验记录：推理系统保存历史键和值，以减少重复计算。性能实验应固定输入输出长度。\n',a.prefix_length)
        # Unique ordinary marker tokens prevent accidental suffix-block reuse.
        excluded=set(llm.tokenizer.all_special_ids)
        markers=[i for i in range(100,100+4*a.batch+100) if i not in excluded][:2*a.batch]
        suffix=exact('请根据记录简要说明实验结论，并解释原因。\n',a.input_length-a.prefix_length-1)
        batches=[[prefix+[markers[phase*a.batch+i]]+suffix for i in range(a.batch)] for phase in range(2)]
        sp=SamplingParams(temperature=0,max_tokens=a.output_length,ignore_eos=True)
        # Warm complete full-prefill/decode shapes, then clear reusable metadata.
        llm.generate(batches[0],sp,use_tqdm=False)
        llm.generate(batches[1],sp,use_tqdm=False)
        bm=llm.scheduler.block_manager
        assert not bm.used_block_ids
        bm.hash_to_block_id.clear()
        for block in bm.blocks: block.hash=-1; block.token_ids=[]
        result=dict(batch=a.batch,repeat=a.repeat,cache=a.cache,environment=environment,
                    workload=dict(input_tokens=a.input_length,prefix_tokens=a.prefix_length,output_tokens=a.output_length))
        for phase,prompts in zip(('cold','warm'),batches):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            outputs=llm.generate(prompts,sp,use_tqdm=False)
            torch.cuda.synchronize(); seconds=time.perf_counter()-started
            ids=[o['token_ids'] for o in outputs]; stats=dict(llm.last_batch_stats)
            assert len(ids)==a.batch and all(len(x)==a.output_length for x in ids)
            assert not stats['preemptions'] and stats['used_blocks']==0
            if not a.cache: assert stats['reused_tokens']==0
            if phase=='warm' and a.cache: assert stats['reused_tokens']==a.batch*a.prefix_length
            result[phase]=dict(prompts=prompts,outputs=ids,seconds=seconds,output_tok_s=a.batch*a.output_length/seconds,
                               stats=stats,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        llm.exit()


if __name__=='__main__':
    main()
