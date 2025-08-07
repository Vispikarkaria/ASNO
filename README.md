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

