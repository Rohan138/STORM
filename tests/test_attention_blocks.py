import unittest
from sub_models import attention_blocks

from tinygrad import Tensor


class TestMask(unittest.TestCase):
    def test_vector_mask(self):
        attn = attention_blocks.get_vector_mask(10)
        self.assertEqual(attn.shape, (1, 1, 10))

    def test_subsequent_mask(self):
        seq = Tensor.ones((1, 10, 10))
        attn = attention_blocks.get_subsequent_mask(seq)
        self.assertEqual(attn.shape, (1, 10, 10))

    def test_subsequent_mask_with_batch_length(self):
        attn = attention_blocks.get_subsequent_mask_with_batch_length(10)
        self.assertEqual(attn.shape, (1, 10, 10))


class TestMultiHeadAttention(unittest.TestCase):
    def test_multi_head_attention(self):
        model = attention_blocks.MultiHeadAttention(8, 32, 32, 32, 128)
        x = Tensor.ones((1, 1, 32))
        y = model(x)
        self.assertEqual(y.shape, (1, 1, 32))


if __name__ == "__main__":
    unittest.main()
