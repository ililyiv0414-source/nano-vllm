import copy
import json
import unittest
from experiments.summarize_public import ROOT, summarize


class PublicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'benchmarks/a800-greedy.json').read_text(encoding='utf-8'))

    def test_complete_evidence(self):
        self.assertEqual(len(summarize(self.data)),12)

    def test_reject_wrong_hash_and_missing_row(self):
        changed=copy.deepcopy(self.data)
        changed['records'][0]['outputs_sha256']='wrong'
        with self.assertRaises(ValueError): summarize(changed)
        changed=copy.deepcopy(self.data)
        changed['records'].pop()
        with self.assertRaises(ValueError): summarize(changed)
