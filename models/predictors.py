import torch
import torch.nn as nn

from .attention import LayerNorm
from .utils import sequence_mask, generate_path


class VarianceAdopter(nn.Module):
    def __init__(self, channels, dropout):
        super(VarianceAdopter, self).__init__()
        self.duration_predictor = VariancePredictor(
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )
        self.length_regulator = LengthRegulator()
        self.pitch_conv = nn.Conv1d(1, channels, 1)
        self.pitch_predictor = VariancePredictor(
            channels=channels,
            n_layers=5,
            kernel_size=5,
            dropout=dropout
        )
        self.energy_conv = nn.Conv1d(1, channels, 1)
        self.energy_predictor = VariancePredictor(
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )

    def forward(
        self,
        x,
        x_mask,
        y_mask,
        pitch,
        tgt_pitch,
        energy,
        tgt_energy,
        path
    ):
        dur_pred = torch.relu(self.duration_predictor(x, x_mask))

        x = self.length_regulator(x, path)

        pitch = self.pitch_conv(pitch)
        pitch = self.length_regulator(pitch, path)
        pitch = self.pitch_predictor(x.detach() + pitch, y_mask)

        energy = self.energy_conv(energy)
        energy = self.length_regulator(energy, path)
        energy = self.energy_predictor(x + energy, y_mask)

        x += tgt_pitch + tgt_energy
        return x, (dur_pred, pitch, energy)

    def infer(self, x, x_mask, pitch, energy):
        dur_pred = torch.relu(self.duration_predictor(x, x_mask))
        dur_pred = torch.round(dur_pred) * x_mask
        y_lengths = torch.clamp_min(torch.sum(dur_pred, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x_mask.device)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        path = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(1))

        x = self.length_regulator(x, path)

        pitch = self.pitch_conv(pitch)
        pitch = self.length_regulator(pitch, path)
        pitch = self.pitch_predictor(x + pitch, y_mask)

        energy = self.energy_conv(energy)
        energy = self.length_regulator(energy, path)
        energy = self.energy_predictor(x + energy, y_mask)

        x += pitch + energy
        return x, y_mask


class VariancePredictor(nn.Module):
    def __init__(self, channels, n_layers, kernel_size, dropout):
        super(VariancePredictor, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(channels),
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.SiLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])

        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x)
            x *= x_mask
        x = self.out(x)
        x *= x_mask
        return x


class LengthRegulator(nn.Module):
    def forward(self, x, path):
        x = torch.bmm(x, path)
        return x
