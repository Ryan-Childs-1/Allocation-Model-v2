# Allocation AI — TensorFlow-Free Streamlit Runtime

This package is a deployment-safe version of the two-network Keras FLM ranker app.

## Why this version exists

Streamlit Cloud was failing while trying to install TensorFlow/Keras under Python 3.13. This version avoids that dependency entirely. The trained `.keras` networks were exported into compressed NumPy weight files:

- `allocation_model.npz`
- `deallocation_model.npz`

The app performs the same Dense + BatchNorm + Swish/Sigmoid forward pass in NumPy, so the deployment no longer needs TensorFlow.

## Required files

- `app.py`
- `base_features.py`
- `model_utils.py`
- `allocation_model.npz`
- `deallocation_model.npz`
- `preprocessing_pipeline.joblib`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

Use `app.py` as the entry point. No TensorFlow dependency is required.

## Model method

The app uses two separate neural networks:

1. Allocation network ranks each potential FLM unit.
2. Deallocation network removes FLM units from rows/groups that exceed DC availability.

The final safety pass ensures `AI Left DC >= 0`.
