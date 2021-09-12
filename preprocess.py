from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from resemblyzer import trim_long_silences

from transform import TacotronSTFT
from dtw import dtw

ORIG_SR = None
NEW_SR = None


class PreProcessor:

    def __init__(self, config_path):
        config = OmegaConf.load(config_path)
        self.src_dir = Path(config.src_dir)
        self.tgt_dir = Path(config.tgt_dir)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.to_mel = TacotronSTFT()

        global ORIG_SR, NEW_SR
        ORIG_SR = config.orig_sr
        NEW_SR = config.new_sr

    @staticmethod
    def load_wav(wav_path):
        wav, sr = sf.read(wav_path)
        wav = librosa.resample(wav, ORIG_SR, NEW_SR)
        wav_trimmed = trim_long_silences(wav)
        return wav_trimmed

    @staticmethod
    def extract_feats(wav):
        f0, sp, ap = pw.wav2world(wav, NEW_SR, 1024, 256 / NEW_SR * 1000)
        mfcc = pw.code_spectral_envelope(sp, NEW_SR, 24)
        return f0, sp, ap, mfcc

    def process_speaker(self, dir_path):
        wav_paths = list(sorted(list(dir_path.glob('*.wav'))))

        wavs = list()
        mels = list()
        lengths = list()
        pitches = list()
        energies = list()
        mfccs = list()

        for i in tqdm(range(len(wav_paths))):
            wav = self.load_wav(wav_paths[i])
            pitch, *_, mfcc = self.extract_feats(wav)
            mel, energy = self.to_mel(torch.FloatTensor(wav)[None, :])

            pitch = np.array(pitch).astype(np.float32)
            energy = np.array(energy).astype(np.float32)

            pitch[pitch != 0] = np.log(pitch[pitch != 0])
            energy[energy != 0] = np.log(energy[energy != 0])

            assert pitch.shape[0] == mel.size(-1)

            wavs.append(wav)
            mels.append(mel)
            lengths.append(mel.size(-1))
            pitches.append(pitch)
            energies.append(energy)
            mfccs.append(mfcc)

        return wavs, mels, pitches, energies, mfccs, lengths

    def preprocess(self):
        print('Start Source')
        src_wavs, src_mels, src_pitches, src_energies, src_mfccs, src_lengths = self.process_speaker(self.src_dir)
        print('Start Target')
        tgt_wavs, tgt_mels, tgt_pitches, tgt_energies, tgt_mfccs, tgt_lengths = self.process_speaker(self.tgt_dir)

        assert len(src_mfccs) == len(tgt_mfccs), 'Dataset must be parallel data.'
        print('Calculate DTW')
        paths = [dtw(src_mfccs[i], tgt_mfccs[i], interp=False) for i in tqdm(range(len(src_mfccs)))]

        print('Save file')
        for i in tqdm(range(len(src_mels))):
            torch.save([
                torch.FloatTensor(src_wavs[i])[None, :],
                torch.FloatTensor(tgt_wavs[i])[None, :],
                src_mels[i].squeeze().transpose(0, 1),
                tgt_mels[i].squeeze().transpose(0, 1),
                src_lengths[i],
                tgt_lengths[i],
                torch.FloatTensor(src_pitches[i]).view(-1, 1),
                torch.FloatTensor(tgt_pitches[i]).view(-1, 1),
                torch.FloatTensor(src_energies[i]).view(-1, 1),
                torch.FloatTensor(tgt_energies[i]).view(-1, 1),
                torch.FloatTensor(paths[i])
            ], self.output_dir / f'data_{i+1:04d}.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/preprocess.yaml')
    args = parser.parse_args()
    PreProcessor(args.config).preprocess()
