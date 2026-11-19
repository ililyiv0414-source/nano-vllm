# nano-vllm · Inference Optimization Lab

[Interactive evidence explorer](https://ililyiv0414-source.github.io/nano-vllm/) · [CPU checks](https://github.com/ililyiv0414-source/nano-vllm/actions/workflows/cpu.yml)
**Profile inference. Change one thing. Verify every token.**

[中文](README.zh-CN.md) · [Measured data](benchmarks/a800-greedy.json) · [Contribute](CONTRIBUTING.md)

A hands-on extension of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), focused on prefix-cache diagnostics, CPU/CUDA profiling and an opt-in greedy sampling fast path.

## Measured, not assumed

On **A800, Qwen3-0.6B, BF16, CUDA Graph, 512 input / 64 output tokens**, five alternating-order paired repetitions showed the following median batch latency reductions. Output token IDs matched.

| Batch | Prefix off: cold / warm | Prefix on: cold / warm |
|---|---|---|
| 1 | 6.31% / 6.05% | 5.84% / 4.67% |
| 4 | 5.49% / 5.53% | 5.68% / 5.58% |
| 16 | 4.48% / 4.49% | 4.30% / 4.54% |

Each value is the median of `1 - fast_seconds / baseline_seconds`, not the ratio of independently aggregated times. Both paths use CUDA Graph. Cold/warm observations share workers. This is not a statistical-significance claim, TTFT, online P99, or a guarantee for other workloads. Model loading and graph capture are excluded.

The initial eager pilot showed **no end-to-end improvement**. Enabling upstream CUDA Graph had a separate large benefit; that is not an original optimization here. Synthetic prefix-cache output divergence remains a known diagnostic case, not a universally solved issue.

## Three-step CPU demo

```bash
# 1. Get the code
git clone https://github.com/ililyiv0414-source/nano-vllm.git
cd nano-vllm
# 2. Install dependencies (a virtual environment is recommended)
python -m pip install -r requirements-cpu.txt
# 3. Run
python exercises/kv_cache_cpu.py
```

Expected ending:

```text
PASS: same attention output; historical K/V projections were reused.
Projection token count: full recomputation=55, KV cache=10. Attention still reads past K/V.
```

This single-layer exercise is not a complete model benchmark. No GPU or model download is needed.

## What changed

- An all-greedy batch can skip temperature preparation and unnecessary random sampling, using compiled argmax instead.
- Mixed/random batches retain the original sampler. Enable explicitly with `enable_greedy_fastpath=True`; default is off.
- Cache switches, work counters, output validation, layer tracing and paired experiments make performance claims auditable.
- The fast path changes RNG consumption, so later random requests may differ under the same seed.

Start with [sampler](nanovllm/layers/sampler.py), [runner](nanovllm/engine/model_runner.py), and [benchmark](experiments/benchmark_text.py).

## GPU reproduction

Tested: Linux x86_64, Python 3.10.19, PyTorch 2.6.0+cu124, Triton 3.2.0, FlashAttention 2.7.4.post1, Transformers 4.52.4. Match FlashAttention to Python, PyTorch and C++ ABI. Source compilation requires a compatible CUDA toolkit. Model revision: `c1899de289a04d12100db370d81485cdf75e47ca` (Qwen/Qwen3-0.6B).

With compatible dependencies and local weights:

```bash
python -m pip install --no-deps -e .
python experiments/smoke.py --model /path/to/Qwen3-0.6B
python -m unittest discover -s tests -v
```

Compare identical workloads, alternating baseline/fast order across five repetitions:

```bash
python experiments/benchmark_text.py --model /path/to/Qwen3-0.6B --batches 1 4 16 --repeats 1 --cuda-graph --out results/base-01
python experiments/benchmark_text.py --model /path/to/Qwen3-0.6B --batches 1 4 16 --repeats 1 --cuda-graph --greedy-fastpath --out results/fast-01
```

Use new folders each repetition. The report's off/on columns refer to **prefix caching**; compare matching records across folders to measure the greedy change. Verify input/output IDs and work counters before computing speedup. GPU profiling is separate from benchmarking.

## Attribution

Based on upstream commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`, by Xingkai Yu. The engine, paged cache, prefix reuse, model and CUDA Graph come from upstream. This is not an independently rewritten engine. Extensions and analysis were AI-assisted; the maintainer executed A800 verification. The upstream [MIT license](LICENSE) is retained.

The [public evidence](benchmarks/a800-greedy.json) preserves individual timings, token-array hashes/lengths and work counts while removing private machine paths, UUIDs and environment dumps. Historical source fingerprints and worker hashes are retained.

If this lab helps you, a star helps others discover it. Reproductions and minimal failing cases are especially welcome.
