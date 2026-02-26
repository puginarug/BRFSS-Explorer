# ScreenMind — Predicting Mental Health Risk from Survey Data

> Can we predict who is at risk of poor mental health from a 15-question survey?
> **Spoiler: kind of. And understanding *why* it's hard is the most interesting part.**

**[▶ Live Dashboard](https://YOUR-APP.streamlit.app)** &nbsp;|&nbsp; BRFSS 2024 · 449,000 US adults · PyTorch · W&B

---

## The Finding

Trained a PyTorch neural network on CDC public health survey data (449k respondents)
to predict mental health burden.

**What works:** The classifier achieves **AUC 0.856** — solid for survey data.
Loneliness, lack of emotional support, and life dissatisfaction show the clearest
population-level gradients.

**What's hard:** Precision at the default threshold is only **~35%** — for every
3 people flagged as high-risk, 2 are false alarms. This is the precision-recall
tradeoff in a clinical context, and it's why AUC alone is a misleading headline.

**Why prediction has a ceiling:** Mental health is shaped by hundreds of factors
a 15-variable survey can't see — trauma history, relationships, neurobiology.
Loneliness is the strongest social predictor in this dataset, but it explains
only part of the variance. The rest is genuinely unknowable from this data.

---

## What Was Built

| Component | Description |
|-----------|-------------|
| **EDA** | Full exploratory analysis of BRFSS 2024 — distribution of mental health burden, feature correlations, zero-inflation |
| **Preprocessing pipeline** | Sentinel code cleaning, stratified 70/15/15 split, median imputation, StandardScaler — all fit on training data only |
| **PyTorch MLP** | Two models from scratch: binary classifier (high-risk) + regressor (days). Custom training loop, early stopping, checkpointing |
| **Class imbalance handling** | Weighted BCE loss (`pos_weight ≈ 6.6×`) and custom `WeightedMSELoss` that penalises errors on high-burden cases |
| **W&B experiment tracking** | 3 architecture variants logged with per-epoch loss curves and final metrics to a live dashboard |
| **Interactive EDA app** | Streamlit dashboard — pick any feature as X-axis, see its relationship to mental health days or life satisfaction |

---

## Key Results

**Classifier** (binary: >14 bad days/month = high-risk)

| Metric | Value | Note |
|--------|-------|------|
| ROC-AUC | **0.856** | Best variant |
| Recall | **77.6%** | Fraction of high-risk cases caught |
| Precision | **35.2%** | Of those flagged, actually high-risk |
| F1 | 48.5% | Low F1 is expected with 13% positive rate |

**Regressor** (predict exact days 0–30)

| Metric | Value | Baseline (predict mean) |
|--------|-------|------------------------|
| MAE | **4.17 days** | 5.92 days |
| R² | **0.35** | 0.0 |

---

## Tech Stack

```
Data & ML       Python 3.11 · PyTorch · NumPy · Pandas · scikit-learn (preprocessing only)
Experiment log  Weights & Biases
Dashboard       Streamlit · Plotly · SciPy
Dataset         BRFSS 2024 (CDC) — public domain, 449k US adults
```

---

## Run Locally

```bash
git clone https://github.com/YOUR-USERNAME/ScreenTimeML.git
cd ScreenTimeML
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

The processed test data (~4.5 MB) is included in the repo so the dashboard
works out of the box. The raw BRFSS file (1 GB) and model weights are excluded.

To reproduce training from scratch, download `LLCP2024.XPT` from the
[CDC BRFSS 2024 page](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
and place it in `data/raw/`, then run notebooks `01` through `04` in order.

---

## Project Structure

```
ScreenTimeML/
├── dashboard/
│   └── app.py              ← Streamlit EDA dashboard (run this)
├── notebooks/
│   ├── 01_eda.ipynb        ← Exploratory analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb      ← MLP training + evaluation
│   └── 04_experiments.ipynb ← W&B experiment tracking, 3 variants
├── src/
│   ├── data/preprocessing.py   ← BRFSS cleaning + split + scale
│   ├── models/mlp.py           ← PyTorch MLP (configurable depth/width)
│   └── training/trainer.py     ← Training loop, losses, evaluation
├── data/
│   └── processed/          ← Test split + scaler (committed, ~4.5 MB)
└── requirements.txt
```

---

## About

Built by **David Zingerman** — computational biologist (M.Sc., Weizmann Institute)
transitioning to industry data science.

This project was built end-to-end as a learning exercise: PyTorch from scratch,
real experiment tracking discipline with W&B, and honest evaluation of a genuinely
hard prediction problem. The dataset topic — mental health — is something I care
about, which made the ceiling on prediction accuracy more interesting than frustrating.
