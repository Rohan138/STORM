import unittest
from sub_models import attention_blocks

from tinygrad import Tensor


class TestMask(unittest.TestCase):

    def test_subsequent_mask(self):
        seq = Tensor.ones((1, 10, 10))
        attn = attention_blocks.get_subsequent_mask(seq)
        self.assertEqual(attn.numpy().shape, (1, 10, 10))

    def test_subsequent_mask_with_batch_length(self):
        attn = attention_blocks.get_subsequent_mask_with_batch_length(10)
        self.assertEqual(attn.numpy().shape, (1, 10, 10))


class TestMultiHeadAttention(unittest.TestCase):
    def test_multi_head_attention(self):
        model = attention_blocks.MultiHeadAttention(8, 32, 32, 8, 128)
        x = Tensor.randint((1, 1, 32))
        y = model(x)
        self.assertEqual(y.numpy().shape, (1, 1, 32))
    
    def test_multi_head_attention_kv_cache(self):
        model = attention_blocks.MultiHeadAttention(8, 32, 32, 8, 128)
        x = Tensor.randint((1, 1, 32))
        y = model(x, 0, cache_kv=True)
        self.assertEqual(y.numpy().shape, (1, 1, 32))
        x = Tensor.randint((1, 1, 32))
        y = model(x, 1, cache_kv=True)
        self.assertEqual(y.numpy().shape, (1, 1, 32))

class TestAttentionBlockKVCache(unittest.TestCase):
    def test_attention_block(self):
        model = attention_blocks.AttentionBlockKVCache(32, 32, 8, 128, 0.1)
        x = Tensor.randint((1, 1, 32))
        y = model(x)
        self.assertEqual(y.numpy().shape, (1, 1, 32))

    def test_attention_block_kv_cache(self):
        model = attention_blocks.AttentionBlockKVCache(32, 32, 8, 128, 0.1)
        x = Tensor.randint((1, 1, 32))
        y = model(x, 0, cache_kv=True)
        self.assertEqual(y.numpy().shape, (1, 1, 32))
        x = Tensor.randint((1, 1, 32))
        y = model(x, 1, cache_kv=True)
        self.assertEqual(y.numpy().shape, (1, 1, 32))


if __name__ == "__main__":
    unittest.main()
