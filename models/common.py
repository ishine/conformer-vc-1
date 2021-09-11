import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionModule(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super(ConvolutionModule, self).__init__()
        self.layer_norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels * 2, 1)
        self.glu = GLU(dim=1)
        self.depth_wise_conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.layer_norm(x)
        x = self.conv1(x) * x_mask
        x = self.glu(x)
        x = self.depth_wise_conv(x) * x_mask
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.conv2(x) * x_mask
        x = self.dropout(x)
        return x


class FFN(nn.Module):
    def __init__(self, channels, dropout):
        super(FFN, self).__init__()

        self.norm = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x * x_mask)
        x = self.dropout(x)
        return x * x_mask


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        x = x * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return x


class GLU(nn.Module):
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.glu(x, self.dim)


class RelPositionalEncoding(nn.Module):
    def __init__(self, channels, dropout=0.1, max_len=10000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = channels
        self.scale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(2) >= x.size(2) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.transpose(-1, -2).to(device=x.device, dtype=x.dtype).half()

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            :,
            self.pe.size(2) // 2 - x.size(2) + 1 : self.pe.size(2) // 2 + x.size(2),
        ]
        return x, self.dropout(pos_emb)
