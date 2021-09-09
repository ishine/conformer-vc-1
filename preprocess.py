from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
import torch
from omegaconf import OmegaConf
from resemblyzer import trim_long_silences
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from transform import TacotronSTFT
from dtw import dtw

ORIG_SR = 48000
NEW_SR = 24000


class PreProcessor:

    def __init__(self, config):
        self.jmvd_dir = Path(config.jmvd_dir)
        self.jsut_dir = Path(config.jsut_dir)

        self.label_dir = Path(config.label_dir)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.to_mel = TacotronSTFT()

    def load_jmvd(self, wav_path):
        wav, sr = sf.read(wav_path)
        wav = librosa.resample(wav, ORIG_SR, NEW_SR)
        wav_trimmed = trim_long_silences(wav)
        return wav_trimmed

    @staticmethod
    def get_time(path, sr=24000):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        b, e = lines[1], lines[-2]
        begin_time = int(float(b.split(' ')[0]) * 1e-7 * sr)
        end_time = int(float(e.split(' ')[1]) * 1e-7 * sr)
        return begin_time, end_time

    def load_jsut(self, wav_path, label_path):
        wav, sr = sf.read(wav_path)
        wav = librosa.resample(wav, ORIG_SR, NEW_SR)
        b, e = self.get_time(label_path, sr=NEW_SR)
        wav_trimmed = wav[b:e]
        return wav_trimmed

    def extract_feats(self, wav):
        f0, sp, ap = pw.wav2world(wav, NEW_SR, 1024, 256 / NEW_SR * 1000)
        mfcc = pw.code_spectral_envelope(sp, NEW_SR, 24)
        return f0, sp, ap, mfcc

    def normalize(self, feats, scaler):
        mean = scaler.mean_[0]
        std = scaler.scale_[0]
        for i in range(len(feats)):
            feat = feats[i]
            non_zero_idx = feat != 0
            feat[non_zero_idx] = (feat[non_zero_idx] - mean) / std
            feats[i] = feat
        return feats

    def process_jmvd(self):
        jmvd_paths = list(sorted(list(self.jmvd_dir.glob('*.wav'))))

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        wavs = list()
        mels = list()
        lengths = list()
        pitches = list()
        energies = list()
        mfccs = list()

        for i in tqdm(range(len(jmvd_paths))):
            jmvd_wav = self.load_jmvd(jmvd_paths[i])
            pitch, *_, mfcc = self.extract_feats(jmvd_wav)
            mel, energy = self.to_mel(torch.FloatTensor(jmvd_wav)[None, :])

            pitch = np.array(pitch).astype(np.float32)
            energy = np.array(energy).astype(np.float32)

            pitch[pitch != 0] = np.log(pitch[pitch != 0])
            energy[energy != 0] = np.log(energy[energy != 0])

            # pitch_scaler.partial_fit(pitch[pitch != 0].reshape(-1, 1))
            # energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

            assert pitch.shape[0] == mel.size(-1)

            wavs.append(jmvd_wav)
            mels.append(mel)
            lengths.append(mel.size(-1))
            pitches.append(pitch)
            energies.append(energy)
            mfccs.append(mfcc)

        # pitches = self.normalize(pitches, pitch_scaler)
        # energies = self.normalize(energies, energy_scaler)

        return wavs, mels, pitches, energies, mfccs, lengths

    def process_jsut(self):
        jsut_paths = list(sorted(list(self.jsut_dir.glob('*.wav'))))[:550]
        label_paths = list(sorted(list(self.label_dir.glob('*.lab'))))[:len(jsut_paths)]

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        wavs = list()
        mels = list()
        lengths = list()
        pitches = list()
        energies = list()
        mfccs = list()

        for i in tqdm(range(len(jsut_paths))):
            jsut_wav = self.load_jsut(jsut_paths[i], label_paths[i])
            pitch, *_, mfcc = self.extract_feats(jsut_wav)
            mel, energy = self.to_mel(torch.FloatTensor(jsut_wav)[None, :])

            pitch = np.array(pitch).astype(np.float32)
            energy = np.array(energy).astype(np.float32)

            pitch[pitch != 0] = np.log(pitch[pitch != 0])
            energy[energy != 0] = np.log(energy[energy != 0])

            # pitch_scaler.partial_fit(pitch[pitch != 0].reshape(-1, 1))
            # energy_scaler.partial_fit(energy[energy != 0].reshape(-1, 1))

            assert pitch.shape[0] == mel.size(-1)

            wavs.append(jsut_wav)
            mels.append(mel)
            lengths.append(mel.size(-1))
            pitches.append(pitch)
            energies.append(energy)
            mfccs.append(mfcc)

        # pitches = self.normalize(pitches, pitch_scaler)
        # energies = self.normalize(energies, energy_scaler)

        return wavs, mels, pitches, energies, mfccs, lengths

    def preprocess(self):
        print('Start JMVD')
        src_wavs, src_mels, src_pitches, src_energies, src_mfccs, src_lengths = self.process_jmvd()
        print('Start JSUT')
        tgt_wavs, tgt_mels, tgt_pitches, tgt_energies, tgt_mfccs, tgt_lengths = self.process_jsut()

        assert len(src_mfccs) == len(tgt_mfccs)
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
    config = OmegaConf.load('configs/preprocess.yaml')
    PreProcessor(config).preprocess()
