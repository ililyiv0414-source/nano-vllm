# nano-vllm · 推理优化实验室

[English / 完整复现命令](README.md) · [公开实测数据](benchmarks/a800-greedy.json)

基于上游 nano-vLLM，学习如何定位推理耗时、实现改进并验证输出。不是从零重写的推理引擎。

## 核心成果

新增可切换的全 greedy 采样快路径，跳过温度张量准备与不必要的随机采样；混合批次回退原路径。在 A800、Qwen3-0.6B、BF16、CUDA Graph、512输入/64输出、batch=1/4/16下，各缓存条件五轮配对耗时降幅中位数为 **4.30%–6.31%**，输出token一致。

保留无收益结果：eager初步实验没有观察到端到端改善。原有CUDA Graph的收益单独记录，不算新增采样优化的贡献。数值分歧案例也保留，不宣称普遍正确。

## 三步体验，无需GPU

```bash
git clone https://github.com/ililyiv0414-source/nano-vllm.git
cd nano-vllm
python -m pip install -r requirements-cpu.txt
python exercises/kv_cache_cpu.py
```

建议使用虚拟环境。最后显示PASS，10个输入位置的投影token计数为55对10；这是数学练习，不是完整模型加速比。

## 阅读路线

`Sequence → Scheduler → BlockManager → ModelRunner → Sampler → postprocess`

从 `nanovllm/layers/sampler.py` 的 greedy 方法和 `nanovllm/engine/model_runner.py` 的CPU条件分支开始。快路径不消耗原有随机数，因此后续随机请求的同种子序列可能改变。

引擎和已有机制来自 Xingkai Yu 的上游项目，保留MIT许可。新增代码与分析使用AI辅助，维护者执行了真实GPU实验。欢迎提交复现数据、不同硬件结果与失败案例，见[贡献指南](CONTRIBUTING.md)。
