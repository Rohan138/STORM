import unittest
from sub_models import transformer_model

from tinygrad import Tensor, dtypes


class TestTransformerModel(unittest.TestCase):
    def test_storm_transformer_forward(self):
        stoch_dim = 32
        action_dim = 4
        feat_dim = 32
        num_layers = 1
        num_heads = 8
        max_length = 64
        kv_length = 8
        dropout = 0.1
        model = transformer_model.StochasticTransformerKVCache(
            stoch_dim=stoch_dim,
            action_dim=action_dim,
            feat_dim=feat_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            kv_length=kv_length,
            dropout=dropout,
        )

        samples = Tensor.randn(1, 1, stoch_dim)
        action = Tensor.randint(1, 1, high=action_dim)
        mask = Tensor.ones(1, 1).cast(dtypes.bool)
        output = model.forward(samples, action, mask)
        self.assertEqual(output.shape, (1, 1, feat_dim))

    def test_storm_transformer_forward_B_T(self):
        B = 2
        T = 4
        stoch_dim = 32
        action_dim = 4
        feat_dim = 32
        num_layers = 1
        num_heads = 8
        max_length = 64
        kv_length = 8
        dropout = 0.1
        model = transformer_model.StochasticTransformerKVCache(
            stoch_dim=stoch_dim,
            action_dim=action_dim,
            feat_dim=feat_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            kv_length=kv_length,
            dropout=dropout,
        )

        samples = Tensor.randn(B, T, stoch_dim)
        action = Tensor.randint(B, T, high=action_dim)
        mask = Tensor.ones(T, T).tril().cast(dtypes.bool)
        output = model.forward(samples, action, mask)
        self.assertEqual(output.shape, (B, T, feat_dim))

    def test_storm_transformer_forward_with_kv_cache(self):
        stoch_dim = 32
        action_dim = 4
        feat_dim = 32
        num_layers = 1
        num_heads = 8
        max_length = 64
        kv_length = 8
        dropout = 0.1
        model = transformer_model.StochasticTransformerKVCache(
            stoch_dim=stoch_dim,
            action_dim=action_dim,
            feat_dim=feat_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=max_length,
            kv_length=kv_length,
            dropout=dropout,
        )

        samples = Tensor.randn(1, 1, stoch_dim)
        action = Tensor.randint(1, 1, high=action_dim)
        output = model.forward_with_kv_cache(samples, action)
        self.assertEqual(output.shape, (1, 1, feat_dim))
        samples = Tensor.randn(1, 1, stoch_dim)
        action = Tensor.randint(1, 1, high=action_dim)
        output = model.forward_with_kv_cache(samples, action)
        self.assertEqual(output.shape, (1, 1, feat_dim))

        model.reset_kv_cache()
        samples = Tensor.randn(1, 1, stoch_dim)
        action = Tensor.randint(1, 1, high=action_dim)
        output = model.forward_with_kv_cache(samples, action)
        self.assertEqual(output.shape, (1, 1, feat_dim))


if __name__ == "__main__":
    unittest.main()
