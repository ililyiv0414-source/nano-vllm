"""Save an actual small-model generation before starting the benchmark matrix."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='results/smoke.json')
    args = p.parse_args()
    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, enforce_eager=True, max_model_len=1024, max_num_seqs=4,
              max_num_batched_tokens=1024, gpu_memory_utilization=0.8)
    try:
        prompt = llm.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': '用两句话解释什么是 KV Cache。'}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        outputs = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=64))
        result = dict(prompt=prompt, outputs=outputs, stats=llm.last_batch_stats)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(outputs[0]['text'])
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
