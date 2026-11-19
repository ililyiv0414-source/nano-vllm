"""Replay a measured worker and capture warm logits before greedy divergence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--result', help='Original worker JSON to replay')
    parser.add_argument('--out')
    parser.add_argument('--layers', action='store_true', help='Capture request 0 logical KV after warm prefill')
    parser.add_argument('--layer0', action='store_true', help='Capture first-layer suffix activations')
    parser.add_argument('--compare', nargs=2, metavar=('CACHE_OFF', 'CACHE_ON'))
    args = parser.parse_args()
    import torch
    if args.compare:
        a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in args.compare]
        assert a['prompts'] == b['prompts'], 'Different prompts'
        if a.get('activations') and b.get('activations'):
            assert set(a['activations']) == set(b['activations'])
            for name in a['activations']:
                x, y = a['activations'][name].float(), b['activations'][name].float()
                assert x.shape == y.shape
                d = (x-y).abs()
                print('ACT', name, 'equal', torch.equal(x,y), 'max_abs', d.max().item(),
                      'rms_error', d.square().mean().sqrt().item(), 'reference_rms', x.square().mean().sqrt().item())
        if 'layers' in a and 'layers' in b:
            assert set(a['layers']) == set(b['layers'])
            for name in a['layers']:
                for kind in ('k', 'v'):
                    x, y = a['layers'][name][kind], b['layers'][name][kind]
                    assert x.shape == y.shape
                    for region, part in [('prefix', slice(0, 256)), ('suffix', slice(256, 512))]:
                        delta = (x[part].float() - y[part].float()).abs()
                        print('KV', name, kind, region, 'max_abs', delta.max().item(),
                              'mean_abs', delta.mean().item(), 'equal', torch.equal(x[part], y[part]))
        assert len(a['logits']) == len(b['logits']) == 4
        for step, (x, y) in enumerate(zip(a['logits'], b['logits'])):
            assert x.shape == y.shape
            for row in range(x.shape[0]):
                if any(a['outputs'][row][j] != b['outputs'][row][j] for j in range(step)):
                    print('Different history; skip', step, row)
                    continue
                delta = (x[row] - y[row]).abs()
                print('step', step, 'request', row, 'max_abs', delta.max().item(), 'mean_abs', delta.mean().item())
                for label, values in [('off', x[row]), ('on', y[row])]:
                    scores, ids = values.topk(5)
                    print(label, 'top5', list(zip(ids.tolist(), scores.tolist())), 'margin', (scores[0]-scores[1]).item())
        return
    if not all((args.model, args.result, args.out)):
        parser.error('--model, --result and --out required for replay')
    target = Path(args.out)
    if target.exists():
        parser.error('Output exists; use a new path')
    original = json.loads(Path(args.result).read_text(encoding='utf-8'))
    params = original['parameters']
    torch.manual_seed(params['seed'] + params['repeat'])
    from nanovllm import LLM, SamplingParams
    llm = LLM(args.model, enforce_eager=True, enable_prefix_caching=bool(params['enabled']),
              max_model_len=params['input_length'] + params['output_length'],
              max_num_seqs=params['batch_size'],
              max_num_batched_tokens=params['batch_size'] * params['input_length'],
              gpu_memory_utilization=0.8)
    try:
        sp = SamplingParams(temperature=0, max_tokens=params['output_length'], ignore_eos=True)
        llm.generate([[1, 2, 3, 4]] * params['batch_size'], sp, use_tqdm=False)
        bm = llm.scheduler.block_manager
        assert not bm.used_block_ids
        bm.hash_to_block_id.clear()
        for block in bm.blocks:
            block.hash = -1
            block.token_ids = []
        cold = next(r for r in original['records'] if r['phase'] == 'cold')
        warm = next(r for r in original['records'] if r['phase'] == 'warm')
        cold_outputs = llm.generate(cold['prompts'], sp, use_tqdm=False)
        print('Cold replay matches:', [o['token_ids'] for o in cold_outputs] == cold['outputs'])
        captured = []
        layers = {}
        activations = {}
        if args.layer0:
            from nanovllm.utils.context import get_context
            first = llm.model_runner.model.model.layers[0]
            def save(name, tensor):
                context = get_context()
                if context.is_prefill and not captured:
                    end = int(context.cu_seqlens_q[1].item())
                    assert end in (256, 512)
                    activations[name] = tensor[end-256:end].detach().cpu().clone()
            def attention_input(module, inputs):
                for name, value in zip(('q', 'k', 'v'), inputs):
                    save(name, value)
            def attention_output(module, inputs, output):
                save('attention_output', output)
            def projection_output(module, inputs, output):
                save('projection_output', output)
            def layer_output(module, inputs, output):
                save('layer_hidden', output[0])
                save('layer_residual', output[1])
            first.self_attn.attn.register_forward_pre_hook(attention_input)
            first.self_attn.attn.register_forward_hook(attention_output)
            first.self_attn.o_proj.register_forward_hook(projection_output)
            first.register_forward_hook(layer_output)
        if args.layers:
            from nanovllm.layers.attention import Attention
            original_run = llm.model_runner.run
            def capture_layers(seqs, is_prefill):
                result = original_run(seqs, is_prefill)
                if is_prefill and not layers:
                    seq = seqs[0]
                    assert seq.num_prompt_tokens == 512 and seq.num_scheduled_tokens + seq.num_cached_tokens == 512
                    table = seq.block_table
                    for name, module in llm.model_runner.model.named_modules():
                        if isinstance(module, Attention):
                            # Logical order, not physical block order. Capture before postprocess releases anything.
                            layers[name] = {kind: cache[table].reshape(-1, *cache.shape[2:])[:512].detach().cpu()
                                            for kind, cache in [('k', module.k_cache), ('v', module.v_cache)]}
                return result
            llm.model_runner.run = capture_layers
        run_model = llm.model_runner.run_model
        def trace(*a, **kw):
            logits = run_model(*a, **kw)
            if len(captured) < 4:
                captured.append(logits.detach().float().cpu())
            return logits
        llm.model_runner.run_model = trace
        outputs = llm.generate(warm['prompts'], sp, use_tqdm=False)
        ids = [o['token_ids'] for o in outputs]
        print('Warm replay matches:', ids == warm['outputs'])
        print('Warm outputs:', ids)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(prompts=warm['prompts'], outputs=ids, logits=captured,
                        stats=llm.last_batch_stats, layers=layers, activations=activations), target)
        print('Saved diagnostic (not a benchmark):', target)
    finally:
        llm.exit()


if __name__ == '__main__':
    main()
