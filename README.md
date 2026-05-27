# VLTS-Net

PyTorch implementation of **VLTS-Net** for the paper:

> **Visual-Conditioned Low-Rank Target Subspace Decomposition for Audio-Visual Speech Separation**

This repository contains the code accompanying a manuscript submitted to **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)**.

The model architecture is implemented in `vlts_net/models/VLTS.py` and registered as `vlts_net.models.VLTS`. This repository is organized as a GitHub-ready research codebase: obsolete reference models were removed, model names were unified, and the training/evaluation scripts use the VLTS model by default.

## Highlights

- Visual-conditioned low-rank target subspace modeling for audio-visual speech separation
- End-to-end waveform reconstruction with magnitude, phase, complex spectrum, and waveform outputs
- PyTorch Lightning training pipeline
- Configurations for LRS2, LRS3, and VoxCeleb2-style data layouts
- Checkpoint serialization and loading through `VLTS.from_pretrain`

## Paper Information

- **Title:** Visual-Conditioned Low-Rank Target Subspace Decomposition for Audio-Visual Speech Separation
- **Model:** VLTS-Net
- **Venue:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
- **Task:** Audio-visual speech separation

## Repository Structure

```text
configs/
  lrs2_vlts.yaml
  lrs3_vlts.yaml
  voxceleb2_vlts.yaml
data_preprocess/
  process_lrs23.py
  process_vox2.py
vlts_net/
  datas/          Dataset and datamodule definitions
  losses/         Matrix-domain and waveform losses
  metrics/        Evaluation metrics
  models/         VLTS model architecture
  system/         Lightning training module and optimizers
  utils/          Utility functions
  videomodels/    Visual frontend
train_vlts.py     Training entrypoint
evaluate_vlts.py  Evaluation entrypoint
infer_vlts.py     Demo inference pipeline
```

Large datasets, checkpoints, experiment outputs, sample videos, and generated audio/video files are excluded through `.gitignore`.

## Installation

Create an environment with Python 3.9 or newer, then install dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA build, install `torch`, `torchaudio`, and `torchvision` from the official PyTorch index first, then install the remaining requirements.

## Data Preparation

Update the dataset paths in the selected config file:

```yaml
datamodule:
  data_config:
    train_dir: data_preprocess/LRS2/tr
    valid_dir: data_preprocess/LRS2/cv
    test_dir: data_preprocess/LRS2/tt
```

The visual frontend checkpoint path is configured here:

```yaml
videonet:
  videonet_config:
    pretrain: pretrain_zoo/lrw_resnet18_mstcn_adamw_s3.pth.tar
```

## Training

```bash
python train_vlts.py --conf_dir configs/lrs2_vlts.yaml
```

Checkpoints and logs are saved under:

```text
Experiments/checkpoint/<exp_name>/
Experiments/tensorboard_logs/<exp_name>/
```

## Evaluation

```bash
python evaluate_vlts.py --conf_dir configs/lrs2_vlts.yaml
```

The evaluator loads:

```text
Experiments/checkpoint/<exp_name>/best_model.pth
```

and writes metrics to:

```text
Experiments/checkpoint/<exp_name>/results/metrics.csv
```

## Inference

`infer_vlts.py` contains the video-demo pipeline for face detection, mouth ROI extraction, audio extraction, separation, and video muxing. Adjust the checkpoint path, input video path, and output directory inside the script before running it.

```bash
python infer_vlts.py
```

## Loading a Trained Model

```python
from vlts_net.models import VLTS

model = VLTS.from_pretrain("Experiments/checkpoint/LRS2-VLTS/best_model.pth")
model.eval()
```

## Forward Interface

```python
denoised_mag, denoised_phase, est_spec, estimated_wav = model(input_wav, mouth_emb)
```

Expected inputs:

- `input_wav`: `(T)`, `(B, T)`, or `(B, 1, T)`
- `mouth_emb`: `(B, C_v, T_v)` or `(B, C_v, T_v, 1)`

Returned tensors:

- `denoised_mag`: compressed-domain magnitude
- `denoised_phase`: phase estimate
- `est_spec`: uncompressed real/imaginary spectrogram
- `estimated_wav`: separated waveform

## Citation


