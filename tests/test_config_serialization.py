import importlib.util
import json
from pathlib import Path
import unittest

from experiments.prefix_cache import serializable_model_config


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
                     'PyTorch and Transformers required; run on GPU host')
class ConfigSerializationTests(unittest.TestCase):
    def test_precision_roundtrip_preserves_runtime_config(self):
        import torch
        from transformers import AutoConfig, Qwen3Config

        model = Path(__file__).resolve().parents[2] / 'models' / 'Qwen3-0.6B'
        config = (AutoConfig.from_pretrained(model, local_files_only=True)
                  if (model / 'config.json').exists() else Qwen3Config(torch_dtype=torch.bfloat16))
        config.dtype = torch.bfloat16
        # Transformers 4.52 does not normalize this additional runtime field.
        raw = config.to_dict()
        if isinstance(raw.get('dtype'), torch.dtype):
            with self.assertRaises(TypeError):
                json.dumps(raw)
        result = serializable_model_config(config)
        restored = json.loads(json.dumps(result))
        self.assertEqual(restored['dtype'], 'torch.bfloat16')
        self.assertEqual(config.dtype, torch.bfloat16)
        self.assertEqual(restored['hidden_size'], config.hidden_size)

    def test_both_fields_and_unknown_values(self):
        import torch
        from types import SimpleNamespace

        raw = dict(dtype=torch.bfloat16, torch_dtype=torch.float16, unchanged=7)
        config = SimpleNamespace(to_dict=lambda: raw)
        result = serializable_model_config(config)
        self.assertEqual(result, dict(dtype='torch.bfloat16', torch_dtype='torch.float16', unchanged=7))
        self.assertIs(raw['dtype'], torch.bfloat16)
        self.assertIs(raw['torch_dtype'], torch.float16)
        raw['unexpected'] = object()
        with self.assertRaises(TypeError):
            json.dumps(serializable_model_config(config))


if __name__ == '__main__':
    unittest.main()
