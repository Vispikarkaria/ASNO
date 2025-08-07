````markdown
# Attention-based Spatio-Temporal Neural Operator (ASNO)

This repository contains the implementation and supporting materials for the paper **“An Attention-based Spatio-Temporal Neural Operator for Evolving Physics”** by Vispi Karkaria et al. ASNO integrates a Transformer‐based temporal predictor with a nonlocal attention operator to deliver accurate, interpretable, and generalizable spatio‐temporal modeling of physical systems.

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

Many scientific domains—e.g. fluid flow, climate modeling, additive manufacturing—require learning high-dimensional, nonlinear spatio-temporal dynamics. Traditional methods handle spatial and temporal structure separately, making it hard to interpret or generalize under changing conditions. ASNO overcomes these limitations by:

- **Temporal Extrapolation** via a TransformerEncoder (inspired by an Explicit BDF step).  
- **Spatial Correction** via a Nonlocal Attention Operator (NAO) that mimics an IMEX PDE solver.  

This separation yields:
- **Zero-shot generalization** to unseen parameters  
- **Long-term stability** in chaotic regimes  
- **Interpretable attention maps** isolating time vs. space contributions  

---

## Key Contributions

- **ASNO Architecture**: Combines self-attention temporal modeling with a multi‐block nonlocal attention operator.  
- **Broad Benchmarks**: Tested on Lorenz ODEs, Darcy flow, Navier–Stokes, and DED melt-pool data.  
- **Interpretability**: Attention weights reveal the relative influence of historical states vs. spatial couplings.  
- **State-of-the-Art**: Surpasses transformer-only and standard neural operator baselines in both in‐distribution and out‐of‐distribution tests.

---

## Repository Structure

```text
ASNO/
├── AM_ASNO_training.py         # DED melt-pool (additive manufacturing) training pipeline
├── ASNO_Lorenz_Training.py     # Lorenz system temporal extrapolation experiments
├── ASNO_darcy_training.py      # Darcy flow PDE training & evaluation
├── ASNO_training_code.py       # Shared data loaders, model wrappers, and launch utilities
├── Additive Manufacturing/     # (Optional) raw & processed DED datasets or experiment notes
├── VAE.py                      # Variational autoencoder definitions (used in some pre-training steps)
├── utilities4.py               # MatReader, GaussianNormalizer, LpLoss, and other helpers
└── README.md                   # This file
````

* **`AM_ASNO_training.py`**
  End-to-end pipeline for full-field temperature prediction in Directed Energy Deposition melt pools.
* **`ASNO_Lorenz_Training.py`**
  Experiments on the Lorenz chaotic system, demonstrating zero-shot rollouts.
* **`ASNO_darcy_training.py`**
  Benchmark on Darcy flow: training, OOD evaluation, and error rollouts.
* **`ASNO_training_code.py`**
  Utility functions for dataset handling, model instantiation, training loops, and logging.
* **`Additive Manufacturing/`**
  Placeholder for DED data files, CSVs, or Jupyter notebooks documenting AM experiments.
* **`VAE.py`**
  Encoder, Decoder, and VAE\_model classes for any pre-training or anomaly detection workflows.
* **`utilities4.py`**
  Implements `MatReader` (for `.mat` I/O), `GaussianNormalizer`, `LaplaceApproxLoss` (LpLoss), etc.
* **`README.md`**
  Overview, setup, usage, and citation instructions.

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/vispikarkaria/ASNO.git
   cd ASNO
   ```

2. **Create & activate a virtual environment** (highly recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**

   ```bash
   pip install torch numpy scipy matplotlib tqdm
   ```

   *Alternatively, if you have a `requirements.txt`, run:*

   ```bash
   pip install -r requirements.txt
   ```

---

## Dependencies

* **Python** ≥ 3.8
* **PyTorch** ≥ 1.10
* **NumPy** ≥ 1.20
* **SciPy** ≥ 1.6
* **Matplotlib** ≥ 3.3
* **tqdm** ≥ 4.60
* *(Optional)* **CUDA Toolkit** ≥ 10.2 for GPU support


---

## Model Architecture

All core modules live in **`ASNO_training_code.py`**:

1. **`TransformerEncoder`** (temporal module)

   * Linear embedding → learnable positional encoding → stacked `nn.TransformerEncoderLayer` → final linear projection.

2. **`MS_Loss`** (Nonlocal Attention Operator)

   * `r` heads per block, `nb` sequential blocks
   * Q/K projections, scaled-dot attention, residual + LayerNorm
   * Final projection heads mapping token dimension → output grid.

3. **`CombinedModel`**

   * Chains the two: transformer output concatenated with initial state → NAO → final prediction.

Refer to Section 3 and Figures 1–2 of the paper for full mathematical details.

---

## Training

### DED Melt-Pool

```bash
python AM_ASNO_training.py \
  --data-dir "Additive Manufacturing/data" \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3 \
  --output-dir runs/DED_meltpool
```

### Lorenz System

```bash
python ASNO_Lorenz_Training.py \
  --data-dir "data/lorenz" \
  --epochs 100 \
  --batch-size 128 \
  --lr 5e-4 \
  --output-dir runs/lorenz
```

### Darcy Flow

```bash
python ASNO_darcy_training.py \
  --data-dir "data/darcy" \
  --epochs 75 \
  --batch-size 64 \
  --lr 1e-3 \
  --output-dir runs/darcy
```

Each script accepts flags for data paths, hyperparameters, and an `--output-dir` where it writes:

* Model checkpoints (`model_epoch_*.pt`)
* Training logs (`loss.csv`, `learning_curve.png`)
* TensorBoard summaries (if enabled)

## Citation

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

**Lead Author**: Vispi Nevile Karkaria
Email: [vispikarkaria127@gmail.com]
Repo: [https://github.com/vispikarkaria/ASNO]

Pull requests, issues, and feedback are very welcome!

```
```
