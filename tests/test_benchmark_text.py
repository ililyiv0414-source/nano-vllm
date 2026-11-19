import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.benchmark_text import summarize


class BenchmarkReportTests(unittest.TestCase):
    def test_withhold_ratio_when_outputs_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            (folder/'plan.json').write_text(json.dumps(dict(batches=[1],repeats=1)))
            for cache in (0,1):
                row=dict(prompts=[[1]],outputs=[[2+cache]],seconds=1.,output_tok_s=64.)
                (folder/f'b1-r0-c{cache}.json').write_text(json.dumps(dict(batch=1,repeat=0,cache=cache,environment={},cold=row,warm=row)))
            with patch('builtins.print'):
                summarize(folder)
            self.assertIn('WITHHELD: output mismatch',(folder/'REPORT.md').read_text())
            (folder/'b1-r0-c1.json').unlink()
            with self.assertRaisesRegex(AssertionError,'Incomplete'):
                summarize(folder)
