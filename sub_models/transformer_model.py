from tinygrad import Tensor, nn, dtypes


def get_vector_mask(batch_length, device):
    mask = Tensor.ones((1, 1, batch_length), device=device).cast(dtypes.bool)
    return mask


class MultiHeadAttention:
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def __call__(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).reshape(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).reshape(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).reshape(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = Tensor.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout
        )

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().reshape(sz_b, len_q, -1)
        q = Tensor.dropout(self.fc(q), p=self.dropout)
        q = q + residual

        q = self.layer_norm(q)

        return q


class PositionwiseFeedForward:
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = dropout

    def __call__(self, x):
        residual = x

        x = self.w_2(Tensor.relu(self.w_1(x)))
        x = Tensor.dropout(x, p=self.dropout)
        x = x + residual

        x = self.layer_norm(x)

        return x


class AttentionBlockKVCache:
    def __init__(self, feat_dim, hidden_dim, num_heads, dropout):
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def __call__(self, q, k, v, slf_attn_mask=None):
        output, attn = self.slf_attn(q, k, v, mask=slf_attn_mask)
        output = self.pos_ffn(output)
        return output, attn


class PositionalEncoding1D:
    def __init__(self, max_length: int, embed_dim: int):
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def __call__(self, feat):
        pos_emb = self.pos_emb(Tensor.arange(self.max_length, device=feat.device))
        pos_emb = Tensor.repeat(pos_emb, (feat.shape[0], 1, 1))

        feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        pos_emb = self.pos_emb(Tensor.arange(self.max_length, device=feat.device))
        pos_emb = Tensor.repeat(pos_emb, (feat.shape[0], 1, 1))

        feat = feat + pos_emb[:, position : position + 1, :]
        return feat


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
        super().__init__()
        self.action_dim = action_dim
        self.feat_dim = feat_dim

        # mix image_embedding and action
        self.stem = [
            nn.Linear(stoch_dim + action_dim, feat_dim, bias=False),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
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
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.layer_norm = nn.LayerNorm(
            feat_dim, eps=1e-6
        )  # TODO: check if this is necessary

    def __call__(self, samples, action, mask):
        """
        Normal forward pass
        """
        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat([samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding(feats)
        feats = self.layer_norm(feats)

        for layer in self.layer_stack:
            feats = layer(feats, feats, feats, mask)

        return feats

    def reset_kv_cache_list(self, batch_size, dtype):
        """
        Reset self.kv_cache_list
        """
        self.kv_cache_list = []
        for _ in self.layer_stack:
            self.kv_cache_list.append(
                Tensor.zeros(
                    size=(batch_size, 0, self.feat_dim), dtype=dtype, device="cuda"
                )
            )

    def forward_with_kv_cache(self, samples, action):
        """
        Forward pass with kv_cache, cache stored in self.kv_cache_list
        """
        assert samples.shape[1] == 1
        mask = get_vector_mask(self.kv_cache_list[0].shape[1] + 1, samples.device)

        action = Tensor.one_hot(action.cast(dtypes.int), self.action_dim).float()
        feats = Tensor.cat([samples, action], dim=-1).sequential(self.stem)
        feats = self.position_encoding.forward_with_position(
            feats, position=self.kv_cache_list[0].shape[1]
        )
        feats = self.layer_norm(feats)

        for idx, layer in enumerate(self.layer_stack):
            self.kv_cache_list[idx] = Tensor.cat(
                [self.kv_cache_list[idx], feats], dim=1
            )
            feats = layer(feats, self.kv_cache_list[idx], self.kv_cache_list[idx], mask)

        return feats
