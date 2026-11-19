import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from experiments.prefix_cache import make_batches
from experiments.report import summarize


class ExperimentTests(unittest.TestCase):
    def test_exact_prefix_and_disjoint_suffix(self):
        tokenizer = SimpleNamespace(all_special_ids=[0, 1], vocab_size=1000)
        for prefix in (0, 128, 256):
            batches = make_batches(tokenizer, 32, 512, prefix, 42)
            prompts = batches[0] + batches[1]
            self.assertTrue(all(len(p) == 512 for p in prompts))
            self.assertEqual(len({p[prefix] for p in prompts}), 64)
            self.assertTrue(all(p[:prefix] == prompts[0][:prefix] for p in prompts))
            self.assertFalse(any(t in (0, 1) for p in prompts for t in p))
            self.assertEqual(batches, make_batches(tokenizer, 32, 512, prefix, 42))

    def create_results(self, folder):
        for prefix in (0, 128, 256):
            for enabled in (0, 1):
                data = dict(source_sha256='fixture', parameters=dict(repeat=0, prefix=prefix, enabled=enabled,
                    batch_size=2, input_length=512, output_length=8, seed=42, repeats=1),
                    environment={k: 'fixture' for k in ('torch', 'cuda', 'gpu', 'dtype', 'model_config_sha256',
                                                       'model_weights_sha256', 'pip_freeze')},
                    records=[dict(phase=phase, prompts=[[1] * 512] * 2, outputs=[[3] * 8] * 2,
                                  stats=dict(batch_seconds=1., output_tokens_per_second=16., prefix_hit_ratio=0.))
                             for phase in ('cold', 'warm')])
                (folder / f'r0-p{prefix}-cache{enabled}.json').write_text(json.dumps(data))

    def test_report_outputs_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.create_results(folder)
            mpl, plt = MagicMock(), MagicMock()
            mpl.pyplot = plt
            plt.subplots.return_value = (MagicMock(), MagicMock())
            with patch.dict('sys.modules', {'matplotlib': mpl, 'matplotlib.pyplot': plt}):
                summarize(folder)
            self.assertTrue((folder / 'measurements.csv').exists())
            self.assertIn('通过', (folder / 'REPORT.md').read_text(encoding='utf-8'))
            path = folder / 'r0-p256-cache1.json'
            data = json.loads(path.read_text())
            data['records'][1]['outputs'][0][0] = 99
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                summarize(folder)

    def test_rejects_missing_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.create_results(folder)
            (folder / 'r0-p256-cache1.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                summarize(folder)

    def test_rejects_equal_but_truncated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.create_results(folder)
            for path in folder.glob('*.json'):
                data = json.loads(path.read_text())
                for record in data['records']:
                    record['outputs'] = [[3] * 7] * 2
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Workload shape'):
                summarize(folder)
            self.assertFalse((folder / 'REPORT.md').exists())

    def test_rejects_invalid_metrics(self):
        for metric, value in [('batch_seconds', 0), ('batch_seconds', float('nan')),
                              ('prefix_hit_ratio', 1.1), ('output_tokens_per_second', 999)]:
            with self.subTest(metric=metric, value=value), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                self.create_results(folder)
                path = folder / 'r0-p0-cache0.json'
                data = json.loads(path.read_text())
                data['records'][0]['stats'][metric] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    summarize(folder)


if __name__ == '__main__':
    unittest.main()
