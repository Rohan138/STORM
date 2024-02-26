from tinygrad import Tensor, Variable, nn, dtypes
from sub_models.attention_blocks import (
    AttentionBlockKVCache,
    PositionalEncoding1D,
    get_vector_mask,
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
                max_length=max_length,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary
        self.kv_len = Variable("kv_len", 0, max_length)
        self.max_length = max_length

    def forward(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat([samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats = layer(feats, 0, False, mask)

        return feats

    def forward_with_kv_cache(self, samples, action):
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_len + 1, samples.device)

        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat([samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats, position=self.kv_len)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats = layer(feats, self.kv_len, True, mask)

        self.kv_len = Variable("kv_len", self.kv_len.val + 1, self.max_length)

        return feats

    def reset_kv_cache(self):
        # lazy hack; reset cache if needed
        self.kv_len = Variable("kv_len", 0, self.max_length)
