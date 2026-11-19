# Contributing

Run the CPU demo and `python -m unittest discover -s tests -v` after installing `requirements-cpu.txt`. Dependency-dependent skips are expected in CPU-only environments and do not validate GPU execution.

For performance changes include hardware, model revision, dtype, graph/eager mode, lengths, batch size, warmup and cache state. Alternate baseline/modified order, preserve all runs, verify output token IDs and work counters, and keep profiling separate from timing. Negative results are welcome.

For numerical differences provide a minimal input and first divergent position. Do not weaken output checks to make a benchmark pass. Explain which behavior is upstream and which is new; disclose AI assistance where relevant.

Do not upload model weights, private addresses, local paths, credentials or full machine environment dumps. Use sanitized data. Keep pull requests focused with relevant tests.
