from tinygrad import Tensor, Variable, nn, dtypes


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    _, batch_length = seq.shape[:2]
    subsequent_mask = (
        1
        - Tensor.triu(
            Tensor.ones((1, batch_length, batch_length), device=seq.device), k=1
        )
    ).cast(dtypes.bool)
    return subsequent_mask


def get_subsequent_mask_with_batch_length(batch_length, device=None):
    """For masking out the subsequent info."""
    subsequent_mask = (
        1
        - Tensor.triu(Tensor.ones((1, batch_length, batch_length), device=device), k=1)
    ).cast(dtypes.bool)
    return subsequent_mask


class MultiHeadAttention:
    """Multi-Head Attention module with KV cache"""

    def __init__(self, n_head, d_model, d_k, d_v, kv_length, dropout=0.1):
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.kv_length = kv_length
        self.cache_k = None
        self.cache_v = None

    def __call__(self, x, start_pos=0, cache_kv=False, mask=None):
        start_pos = start_pos.val if isinstance(start_pos, Variable) else start_pos
        if start_pos == 0:
            self.cache_k = None
            self.cache_v = None

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, seqlen, _ = x.shape

        residual = x

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x).reshape(sz_b, seqlen, n_head, d_k)
        k = self.w_ks(x).reshape(sz_b, seqlen, n_head, d_k)
        v = self.w_vs(x).reshape(sz_b, seqlen, n_head, d_v)

        if cache_kv:
            # create kv cache
            if self.cache_k is None:
                self.cache_k = Tensor.zeros(
                    sz_b, self.kv_length, self.n_head, self.d_k, dtype=x.dtype
                )

            if self.cache_v is None:
                self.cache_v = Tensor.zeros(
                    sz_b, self.kv_length, self.n_head, self.d_v, dtype=x.dtype
                )

            keys = (
                self.cache_k.shrink((None, (0, start_pos), None, None)).cat(k, dim=1)
                if start_pos > 0
                else k
            )
            values = (
                self.cache_v.shrink((None, (0, start_pos), None, None)).cat(v, dim=1)
                if start_pos > 0
                else v
            )

            # update the kv cache
            new_k = (
                keys.pad((None, (0, self.kv_length - start_pos - seqlen), None, None))
                .contiguous()
                .realize()
            )
            self.cache_k.assign(new_k)
            new_v = (
                values.pad((None, (0, self.kv_length - start_pos - seqlen), None, None))
                .contiguous()
                .realize()
            )
            self.cache_v.assign(new_v)
        else:
            keys = k
            values = v

        # Transpose for attention dot product: b x n x lq x dv
        q, keys, values = (
            q.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
        )

        q = Tensor.scaled_dot_product_attention(
            q, keys, values, attn_mask=mask, dropout_p=self.dropout
        )

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().reshape(sz_b, seqlen, -1)
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
    def __init__(self, feat_dim, hidden_dim, num_heads, kv_length, dropout):
        self.slf_attn = MultiHeadAttention(
            num_heads,
            feat_dim,
            feat_dim // num_heads,
            feat_dim // num_heads,
            kv_length,
            dropout=dropout,
        )
        self.pos_ffn = PositionwiseFeedForward(feat_dim, hidden_dim, dropout=dropout)

    def __call__(self, x, start_pos=0, cache_kv=False, slf_attn_mask=None):
        output = self.slf_attn(x, start_pos, cache_kv, slf_attn_mask)
        output = self.pos_ffn(output)
        return output


class PositionalEncoding1D:
    def __init__(self, max_length: int, embed_dim: int):
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = nn.Embedding(self.max_length, embed_dim)

    def __call__(self, feat, position=None):
        pos_emb = self.pos_emb(
            Tensor.arange(self.max_length, device=feat.device).repeat(
                (feat.shape[0], 1)
            )
        )

        if position is not None:
            assert feat.shape[1] == 1
            feat = feat + pos_emb[:, position : position + 1, :]
        else:
            feat = feat + pos_emb[:, : feat.shape[1], :]
        return feat
