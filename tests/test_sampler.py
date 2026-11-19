import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required; run on GPU host')
class SamplerTests(unittest.TestCase):
    def test_greedy_fastpath_ties_and_dtypes(self):
        import torch
        from nanovllm.layers.sampler import Sampler
        sampler = Sampler()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            logits = torch.tensor([[4., 4., 1.], [-3., -1., -2.]], dtype=dtype)
            original = logits.clone()
            expected = Sampler.forward.__wrapped__(sampler, logits.clone(), torch.zeros(2))
            state = torch.get_rng_state()
            actual = Sampler.greedy.__wrapped__(sampler, logits)
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(actual.tolist(), [0, 1])
            self.assertTrue(torch.equal(logits, original))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_mixed_greedy_and_random(self):
        import torch
        from nanovllm.layers.sampler import Sampler
        sampler = Sampler()
        # Test underlying math without requiring a local compiler toolchain.
        logits = torch.tensor([[1., 4., 2.], [2., 1., 3.]])
        temperatures = torch.tensor([0., 1.])
        output = Sampler.forward.__wrapped__(sampler, logits, temperatures)
        self.assertEqual(output[0].item(), 1)
        self.assertIn(output[1].item(), (0, 1, 2))
        greedy = Sampler.forward.__wrapped__(sampler, logits, torch.zeros(2))
        self.assertEqual(greedy.tolist(), [1, 2])


if __name__ == '__main__':
    unittest.main()
