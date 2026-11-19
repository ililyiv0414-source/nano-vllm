from nanovllm.sampling_params import SamplingParams


def __getattr__(name):
    # Allow CPU-only cache exercises without importing CUDA dependencies.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(name)
