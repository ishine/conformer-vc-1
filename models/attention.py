import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm


class RelativeSelfAttentionLayer(nn.Module):
    def __init__(self, channels, n_heads, dropout):
        super(RelativeSelfAttentionLayer, self).__init__()
        self.norm = LayerNorm(channels)
        self.mha = RelativeMultiHeadAttention(channels, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = self.norm(x)
        x = self.mha(x, x, x, pos_emb, attn_mask)
        x = self.dropout(x)
        return x


class RelativeMultiHeadAttention(nn.Module):

    def __init__(self, channels, num_heads, dropout):
        super(RelativeMultiHeadAttention, self).__init__()
        assert channels % num_heads == 0, "d_model % num_heads should be zero."
        self.channels = channels
        self.inner_channels = channels // num_heads
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(channels)

        self.query_proj = nn.Conv1d(channels, channels, 1)
        self.key_proj = nn.Conv1d(channels, channels, 1)
        self.value_proj = nn.Conv1d(channels, channels, 1)
        self.pos_proj = nn.Conv1d(channels, channels, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.inner_channels))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.inner_channels))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, query, key, value, pos_embedding, mask=None):
        B = value.size(0)

        query = self.query_proj(query).view(B, self.num_heads, self.inner_channels, -1)
        key = self.key_proj(key).view(B, self.num_heads, self.inner_channels, -1)
        value = self.value_proj(value).view(B, self.num_heads, self.inner_channels, -1)

        B_pos = pos_embedding.size(0)
        pos_emb = self.pos_proj(pos_embedding).view(B_pos, self.num_heads, self.inner_channels, -1)

        content_score = torch.matmul((query + self.u_bias[None, :, :, None]).transpose(-1, -2), key)
        pos_score = torch.matmul((query + self.v_bias[None, :, :, None]).transpose(-1, -2), pos_emb)
        pos_score = self.rel_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)

        attn_map = F.softmax(score, -1)
        attn = self.dropout(attn_map)

        context = torch.matmul(value, attn)
        context = context.contiguous().view(B, self.channels, -1)

        return self.out_proj(context)

    @staticmethod
    def rel_shift(x):
        B, H, T1, T2 = x.size()
        zero_pad = torch.zeros((B, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(B, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, :T2 // 2 + 1
        ]
        return x
