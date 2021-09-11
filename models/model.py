import torch.nn as nn

from .common import RelPositionalEncoding, PostNet
from .conformer import Conformer
from .predictors import VarianceAdopter
from .utils import sequence_mask


class ConformerVC(nn.Module):
    def __init__(self, params):
        super(ConformerVC, self).__init__()

        self.in_conv = nn.Conv1d(80, params.encoder.channels, 1)
        self.relive_pos_emb = RelPositionalEncoding(
            params.encoder.channels,
            params.encoder.dropout
        )
        self.encoder = Conformer(**params.encoder)
        self.variance_adopter = VarianceAdopter(
            channels=params.encoder.channels,
            dropout=params.encoder.dropout
        )
        self.decoder = Conformer(**params.decoder)
        self.out_conv = nn.Conv1d(params.decoder.channels, 80, 1)
        self.post_net = PostNet(params.decoder.channels)

    def forward(
        self,
        x,
        x_length,
        y_length,
        pitch,
        tgt_pitch,
        energy,
        tgt_energy,
        path
    ):
        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.in_conv(x) * x_mask
        x, pos_emb = self.relive_pos_emb(x)
        x = self.encoder(x, pos_emb, x_mask)

        x, (dur_pred, pitch_pred, energy_pred) = self.variance_adopter(
            x,
            x_mask,
            y_mask,
            pitch,
            tgt_pitch,
            energy,
            tgt_energy,
            path,
        )
        x, pos_emb = self.relive_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask

        x_post = x + self.post_net(x, y_mask)

        return x, x_post, (dur_pred, pitch_pred, energy_pred)

    def infer(self, x, x_length, pitch, energy):
        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)

        x = self.in_conv(x) * x_mask
        x, pos_emb = self.relive_pos_emb(x)
        x = self.encoder(x, pos_emb, x_mask)

        x, y_mask = self.variance_adopter.infer(
            x,
            x_mask,
            pitch,
            energy,
        )
        x, pos_emb = self.relive_pos_emb(x)
        x = self.decoder(x, pos_emb, y_mask)
        x = self.out_conv(x)
        x *= y_mask

        x = x + self.post_net(x, y_mask)
        x *= y_mask
        return x
