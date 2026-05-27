# VLTS-Net

Pre-release PyTorch code for:

> **Visual-Conditioned Low-Rank Target Subspace Decomposition for Audio-Visual Speech Separation**

VLTS-Net formulates audio-visual speech separation as a visual-conditioned low-rank target subspace decomposition problem. The visual stream estimates a compact target-support basis, while the acoustic stream preserves mixture-derived details for magnitude-phase reconstruction.

The main model is implemented in:

```text
vlts_net/models/VLTS.py
```

## Repository Overview

```text
configs/          Example configuration files
data_preprocess/  Dataset preprocessing utilities
vlts_net/         Model, data, loss, metric, and training modules
train_vlts.py     Training entrypoint
evaluate_vlts.py  Evaluation entrypoint
infer_vlts.py     Demo inference script
```

Large datasets, checkpoints, generated audio/video files, and experiment outputs are not included.

## Basic Setup

```bash
pip install -r requirements.txt
```

Please adjust dataset paths, pretrained visual frontend paths, and experiment settings in `configs/` before running experiments.

## Model Usage

```python
from vlts_net.models import VLTS

model = VLTS()
```

The forward interface is:

```python
denoised_mag, denoised_phase, est_spec, estimated_wav = model(input_wav, mouth_emb)
```

## Note

This repository is provided as an early code release. Documentation, checkpoints, and exact reproduction instructions may be updated later.
