# DOCOF: Decomposed Object-Centric Occupancy Forecasting for Stable Long-Horizon BEV Prediction

Official implementation of **DOCOF**, a lightweight object-centric state-space framework for future Bird's-Eye View (BEV) occupancy forecasting.

<p align="center">
  <img src="assets/docof_architecture.png" width="95%">
</p>

## Overview

Future BEV occupancy forecasting is commonly formulated as dense spatio-temporal regression in latent feature space. While effective for short horizons, dense autoregressive forecasting often suffers from error accumulation and unstable long-horizon predictions.

DOCOF reformulates BEV forecasting as a structured state-space evolution problem. Instead of directly predicting future occupancy grids, the framework decomposes forecasting into:

1. Object-centric state extraction
2. Temporal state evolution under bounded dynamics
3. Differentiable occupancy rendering

This design enables interpretable forecasting, improved temporal stability, and extremely lightweight deployment.

---

## Key Features

* Object-centric forecasting paradigm
* Structured state-space dynamics
* Physics-aware motion evolution
* Differentiable occupancy rendering
* Multi-horizon occupancy prediction
* Lightweight forecasting module (0.15M parameters)
* Real-time inference (118 FPS)
* Stability-aware forecasting formulation

---

## Architecture

The overall pipeline consists of three stages:

### Perception Layer

* Multi-camera image encoding
* BEV feature construction
* Object state extraction

### Temporal Dynamics Layer

* Per-object GRU encoder
* Structured state-space evolution
* Physics-aware motion propagation

### Occupancy Forecasting Layer

* Differentiable occupancy renderer
* Multi-horizon BEV prediction

<p align="center">
  <img src="assets/architecture.png" width="100%">
</p>

---

## Prediction Example

<p align="center">
  <img src="assets/demo.gif" width="90%">
</p>

The model predicts future BEV occupancy maps across multiple horizons (t+1, t+2, t+3, t+4).

---

## Repository Structure

```text
DOCOF/
│
├── data/
│   └── loader.py
│
├── models/
│   ├── segmentation.py
│   ├── temporal.py
│   └── decision.py
│
├── losses/
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── utils/
│   ├── metrics.py
│   └── visualization.py
│
├── experiments/
│
├── demo/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your_username/DOCOF.git

cd DOCOF

pip install -r requirements.txt
```

---

## Dataset Preparation

This project uses the nuScenes dataset.

Download the dataset from:

https://www.nuscenes.org

Prepare the dataset according to the structure expected by `data/loader.py`.

---

## Training

```bash
python main.py
```

or

```bash
python training/train.py
```

---

## Evaluation

```bash
python training/evaluate.py
```

---

## Experimental Results

| Metric     | Value  |
| ---------- | ------ |
| Mean IoU   | 43.02  |
| Mean FDE   | 5.10   |
| Parameters | 0.15 M |
| FLOPs      | 0.03 G |
| FPS        | 118    |

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{chakraborty2026docof,
  title={DOCOF: Decomposed Object-Centric Occupancy Forecasting for Stable Long-Horizon BEV Prediction},
  author={Chakraborty, Pritam and others},
  journal={ECCV},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work builds upon ideas from BEV forecasting and autonomous driving literature, including FIERY, ST-P3, BEVFormer, UniAD, and related BEV perception frameworks.
