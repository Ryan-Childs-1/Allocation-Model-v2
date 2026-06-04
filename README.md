# Deployment fix

This package includes `runtime.txt` pinned to `python-3.11` and a Streamlit Cloud-safe `requirements.txt`. This avoids Python 3.14 source builds for `scikit-learn` and TensorFlow/Keras install incompatibilities.

# Allocation AI — Keras FLM Ranker Streamlit App

This is the final Streamlit runtime app built around two separate Keras neural networks:

1. `allocation_model.keras` — ranks individual FLM units from most-deserving to least-deserving.
2. `deallocation_model.keras` — removes FLM units when the first allocation pass exceeds DC availability.

The app accepts `.xlsb`, `.xlsx`, `.xls`, and `.csv` allocation files and outputs a scored file with:

- `AI Starting Left DC`
- `AI Allocation Before Deallocation`
- `AI Deallocation Units`
- `AI DC Safety Cut`
- `AI Final Alloc`
- `AI Left DC`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

Upload all files in this folder to one flat GitHub repo and deploy `app.py`.

## Included diagnostics

The app includes tabs for:

- Running the allocation/deallocation models on a new file
- Execution insights after scoring
- Training metrics and epoch logs
- Feature insight and candidate names
- Keras model architecture
- Cutoff sweep and validation evaluation

## Required artifacts

These files must stay next to `app.py`:

- `allocation_model.keras`
- `deallocation_model.keras`
- `preprocessing_pipeline.joblib`
- `base_features.py`
- `model_utils.py`
