"""Small causal attention exercise: full recomputation versus incremental KV cache."""
import numpy as np


def attention(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True) @ v


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(10, 4))
    wq, wk, wv = [rng.normal(size=(4, 4)) for _ in range(3)]
    keys, values = [], []
    for t in range(len(x)):
        reference = attention(x[t:t+1] @ wq, x[:t+1] @ wk, x[:t+1] @ wv)
        keys.append(x[t] @ wk)
        values.append(x[t] @ wv)
        cached = attention(x[t:t+1] @ wq, np.array(keys), np.array(values))
        np.testing.assert_allclose(cached, reference, atol=1e-12)
        print(f'step={t+1}, Q=(1,4), K/V=({t+1},4), max_error={np.max(abs(cached-reference)):.2e}')
    print('PASS: same attention output; historical K/V projections were reused.')
    n = len(x)
    full_count = n * (n + 1) // 2
    print(f'Projection token count: full recomputation={full_count}, KV cache={n}. Attention still reads past K/V.')


if __name__ == '__main__':
    main()
