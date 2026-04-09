# CausalML — Uplift Modeling Platform

A 4-step Streamlit app for CATE estimation using S/T/X/R meta-learners via the official `causalml` library.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
├── app.py                  # Streamlit app (presentation layer)
├── streamlit_app/          # Service layer
│   ├── constants.py        # Learner options, column name hints
│   ├── models.py           # Typed dataclasses (RunConfig, TrainingResult …)
│   └── services/
│       ├── causalml_adapter.py  # Instantiates S/T/X/R learners + base models
│       ├── data_io.py           # Loads demo or uploaded CSV data
│       ├── evaluation.py        # AUUC, gain curve, CATE RMSE
│       ├── explanations.py      # Segment summary, feature importance
│       ├── preprocessing.py     # Train/test split, imputation, encoding
│       ├── training.py          # Orchestrates full training pipeline
│       └── validation.py        # Column validation + auto-detection
├── requirements.txt
└── .streamlit/config.toml  # Theme + server config
```

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account, select the repo, set main file to `app.py`
4. Click Deploy
