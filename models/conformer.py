import torch.nn as nn

from .attention import LayerNorm, MultiHeadAttention
from .common import GLU


class ConformerLayer(nn.Module):
    def __init__(self,
                 channels,
                 n_heads,
                 dropout,
                 window_size=4,
                 kernel_size=7):
        super(ConformerLayer, self).__init__()

        self.ff1 = nn.Sequential(
            LayerNorm(channels),
            nn.Conv1d(channels, channels * 4, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 4, channels, 1),
            nn.Dropout(dropout)
        )
        self.norm_pre = LayerNorm(channels)
        self.mha = MultiHeadAttention(channels, channels, n_heads, window_size)
        self.dropout = nn.Dropout(dropout)

        self.conv_module = nn.Sequential(
            LayerNorm(channels),
            nn.Conv1d(channels, channels * 2, 1),
            GLU(dim=1),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout)
        )

        self.ff2 = nn.Sequential(
            LayerNorm(channels),
            nn.Conv1d(channels, channels * 4, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 4, channels, 1),
            nn.Dropout(dropout)
        )

        self.norm_post = LayerNorm(channels)

    def forward(self, x, x_mask):
        x += 0.5 * self.ff1(x)

        residual = x
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = self.norm_pre(x)
        x = self.mha(x, x, attn_mask)
        x = self.dropout(x)
        x += residual

        x += self.conv_module(x)
        x += 0.5 * self.ff2(x)
        x = self.norm_post(x)
        x *= x_mask
        return x


class Conformer(nn.Module):
    def __init__(self,
                 channels=384,
                 n_layers=4,
                 n_heads=2,
                 dropout=0.1):
        super(Conformer, self).__init__()

        self.layers = nn.ModuleList([
            ConformerLayer(
                channels=channels,
                n_heads=n_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x
