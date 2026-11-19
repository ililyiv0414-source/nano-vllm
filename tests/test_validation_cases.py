import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
from experiments.validation_cases import main
from experiments.validation_cases import CASES, build_inputs


class Tokenizer:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class ValidationCasesTests(unittest.TestCase):
    def test_baseline_process_finishes_before_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            out = str(Path(temp) / 'case')
            with patch('sys.argv', ['validation_cases.py', '--model', 'model', '--case', 'boundary257', '--out', out]), patch('experiments.validation_cases.subprocess.run') as run:
                main()
            self.assertEqual(run.call_count, 2)
            first, second = run.call_args_list
            self.assertIn('--baseline-only', first.args[0])
            self.assertTrue(second.args[0][1].endswith('teacher_forced.py'))
            self.assertTrue(first.kwargs['check'])
            self.assertTrue(second.kwargs['check'])

    def test_workload_boundaries(self):
        for name, (length, prefix, batch) in CASES.items():
            with self.subTest(case=name):
                cold, warm = build_inputs(Tokenizer(), length, prefix, batch)
                prompts = cold + warm
                self.assertEqual(len(prompts), batch * 2)
                self.assertTrue(all(len(x) == length for x in prompts))
                self.assertTrue(all(x[:prefix] == prompts[0][:prefix] for x in prompts))
                self.assertEqual(len({x[prefix] for x in prompts}), batch * 2)
                self.assertEqual([cold, warm], build_inputs(Tokenizer(), length, prefix, batch))
