import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class VCDataset(Dataset):
    def __init__(self, fns):
        self.fns = fns

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        return torch.load(self.fns[idx])[2:]


def collate_fn(batch):
    (
        src_mel,
        tgt_mel,
        src_length,
        tgt_length,
        src_pitch,
        tgt_pitch,
        src_energy,
        tgt_energy,
        path
    ) = tuple(zip(*batch))

    src_mel = pad_sequence(src_mel, batch_first=True).transpose(-1, -2)
    tgt_mel = pad_sequence(tgt_mel, batch_first=True).transpose(-1, -2)
    src_pitch = pad_sequence(src_pitch, batch_first=True).transpose(-1, -2)
    tgt_pitch = pad_sequence(tgt_pitch, batch_first=True).transpose(-1, -2)
    src_energy = pad_sequence(src_energy, batch_first=True).transpose(-1, -2)
    tgt_energy = pad_sequence(tgt_energy, batch_first=True).transpose(-1, -2)
    src_length = torch.LongTensor(src_length)
    tgt_length = torch.LongTensor(tgt_length)

    src_max_length = src_length.max()
    tgt_max_length = tgt_length.max()
    path = torch.stack([F.pad(a, (0, tgt_max_length-a.size(-1), 0, src_max_length-a.size(-2))) for a in path], dim=0)

    tgt_duration = path.sum(dim=-1).unsqueeze(1)

    return (
        src_mel,
        tgt_mel,
        src_length,
        tgt_length,
        tgt_duration,
        src_pitch,
        tgt_pitch,
        src_energy,
        tgt_energy,
        path
    )
