```markdown
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

- **Temporal Extrapolation** using a Transformer Encoder, inspired by the Explicit step of a Backward Differentiation Formula (BDF).  
- **Spatial Correction** via a Nonlocal Attention Operator (NAO) that acts as an implicit-explicit solver for static PDE interactions.  

This separable IMEX-inspired design yields zero-shot generalization to unseen physical parameters, long-term stability in chaotic regimes, and interpretable attention maps isolating temporal and spatial contributions.

---

## Key Contributions

- **ASNO Architecture**: Combines self-attention temporal modeling with a nonlocal attention-based neural operator to disentangle temporal and spatial effects.  
- **Zero-Shot Generalization**: Demonstrated on benchmarks including Lorenz ODEs, Darcy flow, Navier–Stokes PDEs, and additive-manufacturing melt pools.  
- **Interpretability**: Attention scores reveal how historical states and external loadings contribute to predictions.  
- **Performance**: Outperforms transformer-only, neural-operator, and hybrid baselines in both in-distribution and out-of-distribution settings.  

---

## Repository Structure

```

ASNO/
├── AM\_ASNO\_training.py            # Additive-manufacturing (DED melt-pool) training script
├── ASNO\_Lorenz\_Training.py        # Lorenz ODE temporal extrapolation experiments
├── ASNO\_darcy\_training.py         # Darcy flow PDE training and evaluation
├── ASNO\_training\_code.py          # Shared utilities: data loaders, model wrappers, training loops
├── Additive Manufacturing/        # (Optional) raw/processed DED data & experiment notes
├── VAE.py                         # Variational Autoencoder (Encoder, Decoder, VAE\_model)
├── utilities4.py                  # MatReader, GaussianNormalizer, LpLoss, and other helpers
└── README.md                      # This file

````

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/ASNO.git
   cd ASNO
````

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

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

> **Note**: You must acquire or generate your own datasets. Place them under `data/` (create this folder if needed) in the following subdirectories:

* **`data/darcy/`**:

  * `.mat` files with fields `'temperature'` (shape `[N, T, S, S]`) and `'input_data'`.
* **`data/lorenz/`**:

  * NumPy arrays or tensors of shape `[N, T, state_dim]` plus optional forcing data.
* **`data/nvs/`**:

  * Serialized time-series of vorticity or velocity fields on a 2D mesh.
* **`data/meltpool/`**:

  * DED melt-pool `.mat` or `.npz` containing laser parameters and temperature fields.

Adjust file paths in each training script or in `ASNO_training_code.py` to match your local layout.

---

## Model Architecture

All core classes live in `ASNO_training_code.py`:

1. **`TransformerEncoder`** (Temporal Extrapolation)

   * Projects inputs → adds learnable positional encoding → stacked `nn.TransformerEncoderLayer` → projects to output.

2. **`MS_Loss`** / **`NAO`** (Spatial Neural Operator)

   * Multi-block, multi-head nonlocal attention with LayerNorm → final output heads per block.

3. **`CombinedModel`**

   * Chains Transformer → concatenates initial state → NAO

Refer to Section 3 and Figures 1–2 in the paper for detailed equations and pseudocode.

---

## Training

### Additive Manufacturing (DED Melt Pool)

```bash
python AM_ASNO_training.py \
  --data-dir data/meltpool \
  --output-dir experiments/am_run1 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3
```

### Lorenz System

```bash
python ASNO_Lorenz_Training.py \
  --data-dir data/lorenz \
  --output-dir experiments/lorenz_run1 \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-4
```

### Darcy Flow

```bash
python ASNO_darcy_training.py \
  --data-dir data/darcy \
  --output-dir experiments/darcy_run1 \
  --epochs 75 \
  --batch-size 16 \
  --lr 2e-3
```

Each script supports flags for learning rate, scheduler, weight decay, and checkpoint frequency. Check the top of each file for full argument lists.

---

## Evaluation

Use the same training scripts with `--evaluate` or run a dedicated evaluation to produce:

* **In-distribution test MSE**
* **Out-of-distribution (OOD) performance**
* **Rollout / multi-step error plots**

Outputs (metrics JSON, PNG plots, checkpoint comparisons) land in your `--output-dir`.

---

## Results & Benchmarks

Precomputed logs and model weights for each benchmark live in `experiments/`. Summaries:

| Benchmark     | Params  | GPU (MB) | Test Loss  | OOD Losses                |
| ------------- | ------- | -------- | ---------- | ------------------------- |
| Darcy Flow    | \~0.76M | \~180    | 0.0368     | 0.0673 / 0.0982 (fwd/bwd) |
| Lorenz System | \~0.26M | \~76     | 0.000794   | —                         |
| Navier–Stokes | \~4.66M | \~880    | 0.0213     | —                         |
| DED Melt Pool | \~5.30M | \~1024   | MAPE 2.50% | —                         |

For complete tables and comparisons, see Tables 1–3 in the paper.

---

## Usage Examples

```python
from ASNO_training_code import CombinedModel, load_checkpoint, visualize_attention

# Load a trained model
model = CombinedModel(...)
model.load_state_dict(torch.load('experiments/darcy_run1/checkpoints/model.pt'))

# Visualize spatial attention on a sample
attention_map = visualize_attention(model, sample_input)
plt.imshow(attention_map)
plt.colorbar()
plt.show()
```

---

## Citation

If you use this code, please cite:

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

**Vispi Nevile Karkaria**
Email: [vnk3019@northwestern.edu](mailto:vnk3019@northwestern.edu)
Repo: [https://github.com/yourusername/ASNO](https://github.com/yourusername/ASNO)

Pull requests and issues are very welcome!
