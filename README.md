# Allocation AI — Keras FLM Ranker Streamlit App

This build is fixed for Streamlit Community Cloud environments that default to Python 3.13.

## Why this version exists

The previous app pinned `tensorflow-cpu==2.16.1` and `numpy<2.0`. Streamlit Cloud ran Python 3.13, where those pins do not have compatible wheels. That forced package builds from source or made TensorFlow unsatisfiable.

This package uses Python 3.13-compatible dependency pins:

- `tensorflow-cpu>=2.21,<2.22`
- `numpy>=2.1,<2.3`
- `scikit-learn>=1.7,<1.8`
- `pandas>=2.2.3,<2.4`

## Files required

- `app.py`
- `base_features.py`
- `model_utils.py`
- `allocation_model.keras`
- `deallocation_model.keras`
- `preprocessing_pipeline.joblib`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud notes

If Streamlit Cloud still selects a different Python version, delete and redeploy the app and choose Python 3.13 or Python 3.11 in Advanced settings. This package is designed to work with Python 3.13.

The app accepts `.xlsb`, `.xlsx`, `.xls`, or `.csv` allocation files and outputs AI allocation/deallocation columns while forcing `AI Left DC >= 0`.
