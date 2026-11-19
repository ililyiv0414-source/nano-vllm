import unittest
from experiments.natural_generation import termination


class TerminationTests(unittest.TestCase):
    def test_eos_and_length(self):
        self.assertEqual(termination([2, 9], 9, 4), 'eos')
        self.assertEqual(termination([2, 9], 9, 2), 'eos')
        self.assertEqual(termination([2], 9, 1), 'length')
        self.assertEqual(termination([2], 9, 4), 'unexpected_short')

    def test_reject_invalid_termination(self):
        for ids, limit in [([], 4), ([9, 2], 4), ([2, 3], 1)]:
            with self.subTest(ids=ids), self.assertRaises(AssertionError):
                termination(ids, 9, limit)
