from tinygrad import Tensor, Variable, nn, dtypes
from sub_models.attention_blocks import (
    AttentionBlockKVCache,
    PositionalEncoding1D,
)


class StochasticTransformerKVCache:
    def __init__(
        self,
        stoch_dim,
        action_dim,
        feat_dim,
        num_layers,
        num_heads,
        max_length,
        kv_length,
        dropout,
    ):
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = [
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            lambda x: Tensor.relu(x),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
        ]
        self.position_encoding = PositionalEncoding1D(
            max_length=max_length, embed_dim=feat_dim
        )
        self.layer_stack = [
            AttentionBlockKVCache(
                feat_dim=feat_dim,
                hidden_dim=feat_dim * 2,
                num_heads=num_heads,
                kv_length=kv_length,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary
        self.kv_idx = Variable("kv_idx", 0, kv_length).bind(0)

        self.reset_kv_cache()

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat(samples, action, dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats = layer(feats, 0, False, mask)

        return feats

    def forward_with_kv_cache(self, samples, action):
        assert samples.shape[1] == 1

        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat(samples, action, dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats, position=self.kv_idx.val)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats = layer(feats, self.kv_idx, True)

        var, val = self.kv_idx.unbind()
        self.kv_idx = var.bind(val + 1)

        return feats

    def reset_kv_cache(self):
        # lazy hack; reset cache if needed
        var, _ = self.kv_idx.unbind()
        self.kv_idx = var.bind(0)
