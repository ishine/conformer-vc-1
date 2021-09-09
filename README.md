# Conformer-VC

Conformer-VC is inspired by [Non-autoregressive sequence-to-sequence voice conversion](https://arxiv.org/abs/2104.06793) that is parallel voice conversion methods by using conformer.

The differences between original paper are

- NOT using reduction-factor.
- Mel-spectrograms are not normalized by speaker statistics.
- Use HiFi-GAN instead of PrallelWaveGAN


# Usage

1. Preprocess
Note that this projects is for jmvd dataset and jsut dataset.  
So, if you wanna train your dataset, please rewrite configs/preprocess.yaml and preprocess.py properly.

```bash
# setup for dtw module
$ cd dtw && python setup.py build_ext --inplace && cd ..
$ python prerprocess.py
```

2. Training

```bash
$ ln -s ./dataaset/feats DATA
$ python train.py
```

3. Validation

```bash
$ python validate.py --model_dir {MODEL_DIR} --hifi_gan {HIFI_GAN_DIR} --data_dir DATA
```

if this script run correctly, outputs directory is generated and synthesized wav is in it.
