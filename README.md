# Learning to Hedge Swaptions

Code accompanying the paper:

> **Learning to Hedge Swaptions**  
> Zaniar Ahmadi and Frédéric Godin  
> Concordia University, Quantact/CRM  
> November 26, 2025

This repository implements a deep hedging framework for European swaptions based on reinforcement learning.  
The code compares **deep hedging agents** (trained under different risk objectives) with traditional **rho-hedging** in a three-factor discrete-time arbitrage-free Nelson–Siegel (DTAFNS) model.

The experiments show that:

- Deep hedging achieves **multi-period optimal** dynamic hedges.
- Using **two payer swaps** as hedging instruments is essentially **near-optimal** in this setting.
- Different risk objectives (MSE, downside risk, CVaR) lead to distinct hedging styles and trade-offs between tracking error, downside risk, and tail protection.

---

## 1. Repository structure

```text
learning-to-hedge-swaptions/
├── src/
│   ├── config.py
│   ├── main.py
│   ├── training.py
│   ├── pricing.py
│   ├── networks.py
│   ├── data_generation.py
│   ├── utils.py
│   ├── evaluation.py
│   ├── DTAFNSModels.py
│   ├── DTAFNS_model_forward_measure.py
│   ├── DTAFNS_zero_coupon_price_multipaths.py
│   ├── DTAFNS_swaption_pricing_forward.py
│   ├── DTAFNS_swap_pricing_forward.py
│   └── __init__.py
├── models/
│   └── pretrained/
│           └──  swaption_network_al60_be180_k25083_forgrad_fastkan.pth
├── docs/
│   └── Learning_to_Hedge_Swaptions.pdf
├── README.md
├── .gitignore
└── requirements.txt
```

### Main components

DTAFNSModels.py, DTAFNS_model_forward_measure.py,
DTAFNS_zero_coupon_price_multipaths.py,
DTAFNS_swaption_pricing_forward.py, DTAFNS_swap_pricing_forward.py
: Implementation of the discrete-time arbitrage-free Nelson–Siegel model (DTAFNS), zero-coupon bond pricing, swap and swaption pricing, and measure changes (risk-neutral and forward measures).

data_generation.py
: Configuration and scripts to simulate factor paths and generate Monte Carlo datasets (term-structure states, swaption payoffs, hedging features).

pricing.py, networks.py
: Implementation and training of the Kolmogorov–Arnold Network (KAN) used as a fast swaption pricing surrogate (“deep pricer”) and its sensitivities (rhos).
The file models/swaption_network_al60_be180_k25083_forgrad_fastkan.pth stores a trained FastKAN swaption pricing network.

training.py, main.py
: Training loop for deep hedging agents:

State: yield-curve factors, time-to-maturity, and current portfolio value.

Action: hedge ratios in one, two, or three payer swaps.

Objective: MSE, downside risk, or CVaR of the terminal hedging error.

Optimization via stochastic gradient descent / Adam.

evaluation.py
: Scripts to compute and report the hedging performance metrics on out-of-sample paths:

Mean hedging error

RMSE and root downside risk

CVaR

Probability of under-hedging

Hedging Risk Reduction (HRR)

Trading Intensity (TI)

Dynamic Tracking Error (DTE)

config.py, utils.py
: Configuration utilities, common helper functions, and potentially simple CLI options for running the different experiments.

---

## Installation

Follow the steps below to set up the environment and run the project.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/learning-to-hedge-swaptions.git
cd learning-to-hedge-swaptions
```

### 2. Create and activate a virtual environment (recommended)

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\activate
```
### 3. Install the required Python packages

```bash
pip install -r requirements.txt
```


