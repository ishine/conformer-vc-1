# Conformer-VC

Conformer-VC is inspired by [Non-autoregressive sequence-to-sequence voice conversion](https://arxiv.org/abs/2104.06793) that is parallel voice conversion methods by using conformer.

The differences between original paper are

- NOT using reduction-factor.
- Mel-spectrograms are not normalized by speaker statistics.
- Use HiFi-GAN instead of PrallelWaveGAN

# Requirements

- pytorch
- numpy
- pyworld
- accelerate
- soundfile
- librosa
- cython
- omegaconf
- tqdm
- resemblyzer
- matplotlib
- scipy

If you get an error about the package, please install it.

# Usage

1. Preprocess

If you wanna train your dataset, please rewrite configs/preprocess.yaml and preprocess.py properly.  
Note that num of source files and num of tgt files must be same and file ids must be same.

```bash
$ cd dtw && python setup.py build_ext --inplace && cd ..
$ python prerprocess.py
```

2. Training

single gpu training

```bash
$ ln -s ./dataaset/feats DATA
$ python train.py
```
or multi gpus

```bash
$ ln -s ./dataaset/feats DATA
$ accelerate config

answer question of your machine.

$ accelerate launch train.py
```

3. Validation

```bash
$ python validate.py --model_dir {MODEL_DIR} --hifi_gan {HIFI_GAN_DIR} --data_dir DATA
```

if this script run correctly, outputs directory is generated and synthesized wav is in it.
