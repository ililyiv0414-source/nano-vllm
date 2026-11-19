"""Natural prompts and termination diagnostics; not a performance benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CASES = [
    ('short', '中国的首都是哪座城市？只用两个汉字回答。', 32),
    ('explain', '请用两句话解释大语言模型推理中的 KV Cache，说明缓存了什么。', 96),
    ('arithmetic', '计算 17 加 25。只输出结果，不要解释。', 32),
    ('long_context', '阅读记录：' + '实验使用单张 GPU，输入长度与输出长度分别统计。缓存命中不等于端到端加速。\n' * 12
     + '问题：缓存命中率可以直接当作端到端加速比例吗？用一句话回答。', 64),
    ('limit', '请依次列出数字一到二十，并逐个解释其含义。', 1),
]


def termination(ids, eos, limit):
    assert 0 < len(ids) <= limit, 'Invalid output length'
    assert eos not in ids[:-1], 'Generation continued after EOS'
    return 'eos' if ids[-1] == eos else 'length' if len(ids) == limit else 'unexpected_short'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--backend', choices=('all', 'off', 'on', 'hf'), default='all')
    a = p.parse_args()
    out = Path(a.out)
    if a.backend == 'all':
        if out.exists():
            p.error('Use a new output directory')
        out.mkdir(parents=True)
        for backend in ('off', 'on', 'hf'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--model', a.model,
                            '--out', str(out), '--backend', backend], check=True)
        data = {b: json.loads((out / f'{b}.json').read_text()) for b in ('off', 'on', 'hf')}
        summary = []
        for i, (name, _, _) in enumerate(CASES):
            records = {b: data[b]['records'][i] for b in data}
            assert len({tuple(r['prompt_ids']) for r in records.values()}) == 1
            row = dict(case=name, off_on_equal=records['off']['output_ids']==records['on']['output_ids'],
                       off_hf_equal=records['off']['output_ids']==records['hf']['output_ids'],
                       on_hf_equal=records['on']['output_ids']==records['hf']['output_ids'],
                       results={b: dict(stop=r['stop'], count=len(r['output_ids']), text=r['text']) for b,r in records.items()})
            summary.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Review texts and EOS coverage; no semantic or numerical pass is inferred.')
        return
    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    eos = tokenizer.eos_token_id
    assert isinstance(eos, int)
    prompts = [tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False) for _,text,_ in CASES]
    engine = None
    try:
        if a.backend == 'hf':
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                        torch_dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
        else:
            from nanovllm import LLM, SamplingParams
            engine = LLM(a.model, enforce_eager=True, enable_prefix_caching=a.backend=='on',
                         max_model_len=max(len(x)+c[2] for x,c in zip(prompts,CASES)),
                         max_num_seqs=1, max_num_batched_tokens=max(map(len,prompts)), gpu_memory_utilization=0.8)
        records = []
        for (name, _, limit), prompt in zip(CASES, prompts):
            if engine is not None:
                # Both modes repeat the prompt; only cache-on can reuse prefix KV.
                sp = SamplingParams(temperature=0, max_tokens=limit, ignore_eos=False)
                cold = engine.generate([prompt], sp, use_tqdm=False)[0]['token_ids']
                assert termination(cold, eos, limit) != 'unexpected_short'
                ids = engine.generate([prompt], sp, use_tqdm=False)[0]['token_ids']
                stats = dict(engine.last_batch_stats)
                assert stats['used_blocks'] == 0
            else:
                with torch.inference_mode():
                    inputs = torch.tensor([prompt], device='cuda')
                    output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                do_sample=False, max_new_tokens=limit, eos_token_id=eos,
                                pad_token_id=eos, use_cache=True)
                    ids = output[0, len(prompt):].tolist()
                cold, stats = None, None
            stop = termination(ids, eos, limit)
            assert stop != 'unexpected_short'
            records.append(dict(case=name, prompt_ids=prompt, output_ids=ids, cold_output_ids=cold,
                                limit=limit, eos_token_id=eos, stop=stop,
                                text=tokenizer.decode(ids, skip_special_tokens=True), stats=stats))
            print(a.backend, name, stop, len(ids), flush=True)
        target = out / f'{a.backend}.json'
        assert not target.exists()
        target.write_text(json.dumps(dict(backend=a.backend, records=records), ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if engine is not None:
            engine.exit()


if __name__ == '__main__':
    main()
