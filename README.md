# Attention-based Spatio-Temporal Neural Operator (ASNO)

This repository contains the implementation and supporting materials for the paper **"An Attention-based Spatio-Temporal Neural Operator for Evolving Physics"** by Vispi Karkaria et al. The ASNO framework integrates a Transformer-based temporal predictor with a nonlocal attention-based neural operator to achieve accurate, interpretable, and generalizable modeling of spatio-temporal physical systems.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Dependencies](#dependencies)
6. [Data Preparation](#data-preparation)
7. [Model Architecture](#model-architecture)
8. [Training](#training)
9. [Evaluation](#evaluation)
10. [Results & Benchmarks](#results--benchmarks)
11. [Usage Examples](#usage-examples)
12. [Citation](#citation)
13. [Contact](#contact)

---

## Overview

Modern scientific applications often require modeling high-dimensional, nonlinear spatio-temporal processes (e.g., fluid flow, additive manufacturing). Traditional data-driven architectures capture either spatial or temporal dependencies, but struggle to separate and interpret these combined effects, especially under varying external conditions. ASNO addresses this by:

* **Temporal Extrapolation** using a Transformer Encoder, inspired by the Explicit step of a Backward Differentiation Formula (BDF).
* **Spatial Correction** via a Nonlocal Attention Operator (NAO) that acts as an implicit-exlicit solver for static PDE interactions.

This separable IMEX‐inspired design yields zero-shot generalization to unseen physical parameters, long-term stability in chaotic regimes, and interpretable attention maps isolating temporal and spatial contributions.

---

## Key Contributions

* **ASNO Architecture**: Combines self-attention temporal modeling with a nonlocal attention-based neural operator to disentangle temporal and spatial effects.
* **Zero-Shot Generalization**: Demonstrated on benchmarks including Lorenz ODEs, Darcy flow, Navier–Stokes PDEs, and additive-manufacturing melt pools.
* **Interpretability**: Attention scores reveal how historical states and external loadings contribute to predictions.
* **Performance**: Outperforms transformer-only, neural-operator, and hybrid baselines in both in-distribution and out-of-distribution settings.

---

## Repository Structure

```text
├── data/                  # Placeholder directory for raw and processed datasets
│   ├── darcy/             # Darcy flow datasets
│   ├── lorenz/            # Lorenz system datasets
│   ├── nvs/               # Navier–Stokes datasets
│   └── meltpool/          # DED melt pool temperature datasets
├── src/                   # Source code
│   ├── models.py          # ASNO model definitions (TransformerEncoder, MS_Loss/NAO, CombinedModel)
│   ├── train.py           # Training script with data loading and loops
│   ├── evaluate.py        # Evaluation and metrics scripts
│   ├── utils.py           # Data utilities (MatReader, normalizers, LpLoss)
│   └── config.py          # Hyperparameter configurations
├── experiments/           # Preconfigured experiment logs and checkpoints
├── README.md              # This file
└── requirements.txt       # Python package dependencies
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/asno-framework.git
   cd asno-framework
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dependencies

* Python 3.8+
* PyTorch 1.10+
* NumPy 1.20+
* SciPy 1.6+
* Matplotlib 3.3+
* tqdm 4.60+
* (Optional) CUDA Toolkit 10.2+ for GPU acceleration

All required packages are listed in `requirements.txt`.

---

## Data Preparation

**Note**: The repository provides placeholders. You must download or generate your own datasets matching the formats:

* **Darcy Flow**: structured `.mat` files with `'temperature'` (shape `[N, T, S, S]`) and `'input_data'` fields.
* **Lorenz System**: `.npy` files or tensors of shape `[N, T, state_dim]` plus forcing fields.
* **Navier–Stokes**: serialized time-series of vorticity fields on a 2D mesh.
* **Melt Pool (DED)**: `.mat` or `.npz` containing laser parameters and melt pool temperature fields.

Place processed data under `data/<benchmark>/` and adjust file paths in `config.py` or `train.py` accordingly.

---

## Model Architecture

The core modules are defined in `src/models.py`:

1. **`TransformerEncoder`** (Temporal Extrapolation)

   * Linear input embedding → learnable positional encoding → stack of `nn.TransformerEncoderLayer`
   * Final linear projection to output dimension (e.g., latent dim)

2. **`MS_Loss`** / **`NAO`** (Spatial Neural Operator)

   * Multi-block, multi-head nonlocal attention
   * LayerNorm between blocks
   * Output heads mapping from token dimension to final output grid

3. **`CombinedModel`**

   * Chains Transformer → concatenation with initial state → NAO

Refer to the paper’s Section 3 and Figures 1–2 for detailed equations and pseudocode.

---

## Training

Run the training script:

```bash
python src/train.py \
  --config experiments/config_darcy.yaml \
  --output-dir experiments/darcy_run1
```

Key flags:

* `--config`: YAML file specifying hyperparameters (learning rate, batch size, epochs, model dims).
* `--output-dir`: Directory to save checkpoints, logs, and tensorboard summaries.

Checkpoints are saved every epoch under `<output-dir>/checkpoints/model_epoch_{EPOCH}.pt`. Training curves (loss vs. epoch) are logged via TensorBoard and also saved as PNG/CSV.

---

## Evaluation

To evaluate on a trained checkpoint:

```bash
python src/evaluate.py \
  --checkpoint experiments/darcy_run1/checkpoints/model_epoch_50.pt \
  --data-dir data/darcy \
  --metrics darcy_metrics.json
```

This script computes:

* Test MSE (or chosen norm) on in-distribution data
* OOD performance (if `--ood-config` provided)
* Cumulative rollout error over time

Plots comparing predictions vs. ground truth are saved to `evaluation/`.

---

## Results & Benchmarks

The repository includes precomputed results for all four benchmarks in `experiments/`:

| Benchmark     | Params  | GPU (MB) | Test Loss  | OOD Losses            |
| ------------- | ------- | -------- | ---------- | --------------------- |
| Darcy Flow    | \~0.76M | \~180    | 0.0368     | 0.0673 / 0.0982 (f/b) |
| Lorenz System | \~0.26M | \~76     | 0.000794   | —                     |
| Navier–Stokes | \~4.66M | \~880    | 0.0213     | —                     |
| DED Melt Pool | \~5.3M  | \~1024   | MAPE 2.50% | —                     |

Refer to Tables 1–3 in the paper for detailed comparisons against FNO, U-Net, GNOT, DeepONet, and other baselines.

---

## Usage Examples

### 1. Quick Training on Synthetic Data

```bash
python src/train.py --config experiments/config_synthetic.yaml
```

### 2. Visualize Attention Maps

```python
from src.models import CombinedModel
from src.utils import load_checkpoint, visualize_attention

model = load_checkpoint('checkpoints/model.pt')
attention_map = visualize_attention(model, sample_input)
plt.imshow(attention_map)
plt.colorbar()
```

### 3. Zero-Shot Generalization Demo

```bash
python demos/zero_shot_demo.py --model checkpoints/model.pt \
  --env-config demos/env_configs/new_permeability.yaml
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{karkaria2025asno,
  title={An Attention-based Spatio-Temporal Neural Operator for Evolving Physics},
  author={Karkaria, Vispi and Lee, Doksoo and Chen, Yi-Ping and Yu, Yue and Chen, Wei},
  journal={Under Review},
  year={2025}
}
```

---

## Contact

* **Lead Author**: Vispi Nevile Karkaria ([vnk3019@northwestern.edu](mailto:vnk3019@northwestern.edu))
* **Project Repo**: [https://github.com/yourusername/asno-framework](https://github.com/yourusername/asno-framework)

Pull requests and issues are welcome!
