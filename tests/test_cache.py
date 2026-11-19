import unittest
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams


class CacheTests(unittest.TestCase):
    def setUp(self):
        Sequence.block_size = 256

    def populate(self, bm, tokens):
        seq = Sequence(tokens)
        bm.allocate(seq, bm.can_allocate(seq))
        seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
        bm.hash_blocks(seq)
        bm.deallocate(seq)

    def test_prefix_boundaries_and_last_block(self):
        bm = BlockManager(8, 256)
        self.populate(bm, [1] * 512)
        self.assertEqual(bm.can_allocate(Sequence([1] * 128 + [2] * 384)), 0)
        self.assertEqual(bm.can_allocate(Sequence([1] * 256 + [2] * 256)), 1)
        self.assertEqual(bm.can_allocate(Sequence([1] * 512)), 1)  # leave final block for logits
        self.assertEqual(bm.can_allocate(Sequence([3] * 512)), 0)

    def test_disabled_retains_paging(self):
        bm = BlockManager(3, 256, False)
        for _ in range(20):
            self.populate(bm, [1] * 512)
            self.assertEqual(bm.can_allocate(Sequence([1] * 512)), 0)
            self.assertEqual(len(bm.free_block_ids), 3)
            self.assertFalse(bm.hash_to_block_id)

    def test_shared_refcounts_and_eviction(self):
        bm = BlockManager(4, 256)
        self.populate(bm, [1] * 512)
        a, b = Sequence([1] * 256 + [2] * 256), Sequence([1] * 256 + [3] * 256)
        bm.allocate(a, bm.can_allocate(a))
        bm.allocate(b, bm.can_allocate(b))
        self.assertEqual(a.block_table[0], b.block_table[0])
        self.assertEqual(bm.blocks[a.block_table[0]].ref_count, 2)
        bm.deallocate(a)
        self.assertEqual(bm.blocks[b.block_table[0]].ref_count, 1)
        bm.deallocate(b)
        for i in range(20):
            self.populate(bm, [i + 10] * 512)
            self.assertEqual(len(bm.free_block_ids), 4)
            self.assertFalse(bm.used_block_ids)
            for h, block_id in bm.hash_to_block_id.items():
                self.assertEqual(bm.blocks[block_id].hash, h)

    def scheduler(self, enabled=True, limit=128, blocks=16):
        return Scheduler(SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=limit, eos=-1,
                         kvcache_block_size=256, num_kvcache_blocks=blocks, enable_prefix_caching=enabled))

    def run_batch(self, scheduler, tokens, output=3):
        seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=output))
        scheduler.add(seq)
        steps = 0
        while not scheduler.is_finished():
            seqs, prefill = scheduler.schedule()
            scheduler.postprocess(seqs, [7] * len(seqs), prefill)
            steps += 1
            self.assertLess(steps, 50)
        return seq

    def test_chunked_prefill_and_batch_statistics(self):
        s = self.scheduler()
        self.run_batch(s, [1] * 256 + [2] * 256)
        s.reset_stats()
        seq = self.run_batch(s, [1] * 256 + [3] * 256)
        stats = s.get_stats()
        self.assertEqual(seq.completion_token_ids, [7] * 3)
        self.assertEqual(stats['input_tokens'], 512)
        self.assertEqual(stats['reused_tokens'], 256)
        self.assertEqual(stats['pending_input_tokens'], 256)
        self.assertEqual(stats['prefill_tokens_executed'], 256)
        self.assertEqual(stats['prefix_hit_ratio'], .5)
        self.assertEqual(stats['used_blocks'], 0)

    def test_disabled_decode_and_repeated_batches(self):
        s = self.scheduler(False)
        for _ in range(10):
            s.reset_stats()
            self.run_batch(s, [1] * 512)
            self.assertEqual(s.get_stats()['reused_tokens'], 0)
            self.assertEqual(s.get_stats()['used_blocks'], 0)

    def test_preemption_does_not_double_count_prompt(self):
        s = self.scheduler(limit=512)
        seq = Sequence([1] * 512, SamplingParams(max_tokens=3))
        s.add(seq)
        seqs, prefill = s.schedule()
        s.postprocess(seqs, [7], prefill)
        s.running.remove(seq)
        s.preempt(seq)
        while not s.is_finished():
            seqs, prefill = s.schedule()
            s.postprocess(seqs, [7] * len(seqs), prefill)
        stats = s.get_stats()
        self.assertEqual(stats['input_tokens'], 512)
        self.assertEqual(stats['pending_input_tokens'], 512)
        self.assertEqual(stats['reused_tokens'], 0)
        self.assertEqual(stats['preemptions'], 1)
        self.assertEqual(stats['used_blocks'], 0)

    def test_capacity(self):
        bm = BlockManager(1, 256)
        self.assertEqual(bm.can_allocate(Sequence([1] * 512)), -1)


if __name__ == '__main__':
    unittest.main()
