"""Validate paired greedy outputs before reporting measured performance."""
import csv
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(folder):
    pairs, groups, rows = {}, {}, []
    fingerprints, configurations = set(), set()
    for path in sorted(folder.glob("r*-p*-cache*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        a = data["parameters"]
        env = data["environment"]
        fingerprints.add((data["source_sha256"], env["torch"], env["cuda"], env["gpu"],
                          env["dtype"], env["model_config_sha256"], env["model_weights_sha256"], env["pip_freeze"]))
        configurations.add(tuple(a[k] for k in ('batch_size', 'input_length', 'output_length', 'seed', 'repeats')))
        for record in data["records"]:
            # Equal outputs alone do not prove the requested workload completed.
            for field, length in (("prompts", a["input_length"]), ("outputs", a["output_length"])):
                batch = record[field]
                if len(batch) != a["batch_size"] or any(len(tokens) != length for tokens in batch):
                    raise ValueError(f"Workload shape mismatch: {path.name}: {field}")
                if any(type(token) is not int or token < 0 for tokens in batch for token in tokens):
                    raise ValueError(f"Invalid token IDs: {path.name}: {field}")
            stats = record["stats"]
            for metric in ("batch_seconds", "output_tokens_per_second", "prefix_hit_ratio"):
                value = stats[metric]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid metric: {metric}")
            if stats["batch_seconds"] == 0 or stats["prefix_hit_ratio"] > 1:
                raise ValueError("Invalid metric range")
            expected_throughput = a["batch_size"] * a["output_length"] / stats["batch_seconds"]
            if not math.isclose(stats["output_tokens_per_second"], expected_throughput, rel_tol=1e-6):
                raise ValueError("Throughput inconsistent with completed output tokens")
            key = (a["repeat"], a["prefix"], record["phase"])
            pairs.setdefault(key, {})[a["enabled"]] = (record["prompts"], record["outputs"])
            row = dict(repeat=a["repeat"], prefix=a["prefix"], cache=a["enabled"],
                       phase=record["phase"], **record["stats"])
            rows.append(row)
            groups.setdefault((row["prefix"], row["cache"], row["phase"]), []).append(row)
    if not rows:
        raise ValueError("No measured worker results found")
    if len(fingerprints) != 1 or len(configurations) != 1:
        raise ValueError("Mixed source/environment/model configurations")
    repeats = next(iter(configurations))[-1]
    expected = {(r, p, phase) for r in range(repeats) for p in (0, 128, 256) for phase in ('cold', 'warm')}
    if set(pairs) != expected or len(rows) != len(expected) * 2:
        raise ValueError('Incomplete or duplicated experimental matrix')
    for key, pair in pairs.items():
        if set(pair) != {0, 1} or pair[0] != pair[1]:
            raise ValueError(f"Missing pair or greedy token mismatch at {key}; investigate before performance claims")
    with (folder / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 前缀缓存实测报告", "", "缓存开关的输入及 greedy 输出 Token 对比：通过。",
             "", "耗时包括生成调用及 GPU 同步，不是 TTFT。min/max 表示重复运行范围。",
             "", "| 前缀 | 缓存 | 阶段 | 次数 | 耗时中位数 s [min,max] | 输出 tok/s 中位数 | 命中率中位数 |",
             "|---|---|---|---|---|---|---|"]
    for key, items in sorted(groups.items()):
        ts = [x["batch_seconds"] for x in items]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(items)} | "
                     f"{statistics.median(ts):.4f} [{min(ts):.4f},{max(ts):.4f}] | "
                     f"{statistics.median(x['output_tokens_per_second'] for x in items):.2f} | "
                     f"{statistics.median(x['prefix_hit_ratio'] for x in items):.2%} |")
    lines += ["", "## 分析待填写", "", "- 256 Token 前缀的缓存收益及原因：", "- 128 Token 对照是否符合整块限制：",
              "- 冷批次内是否发生复用，原因是什么：", "- 噪声、缓存淘汰、模型规模与真实业务的差异：",
              "- 个人改动与未完成验证：", "", "显存峰值包含预分配 KV Cache，不能直接解释成有效缓存占用。"]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for enabled in (0, 1):
        keys = sorted(k for k in groups if k[1:] == (enabled, "warm"))
        ax.plot([k[0] for k in keys], [statistics.median(r["batch_seconds"] for r in groups[k]) for k in keys],
                marker="o", label=f"cache={enabled}")
    ax.set(xlabel="Shared prefix (tokens)", ylabel="Warm batch seconds (median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "latency.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
