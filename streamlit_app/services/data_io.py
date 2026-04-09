from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from causalml.dataset.regression import synthetic_data


def file_to_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    del file_name  # included so Streamlit cache invalidates on renamed uploads too
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def load_demo_dataset(n_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    # mode=2 = randomized trial (Nie & Wager 2018 Setup B)
    # Constant propensity=0.5 (no propensity estimation needed) + linear CATE
    # sigma=0.5 keeps noise manageable — AUUC ~0.90 with RF
    np.random.seed(random_state)
    y, X, treatment, tau, _, propensity = synthetic_data(
        mode=2, n=n_rows, p=8, sigma=0.5
    )
    feature_names = [f"feature_{index + 1}" for index in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["treatment"] = treatment.astype(int)
    df["outcome"] = y
    df["true_cate"] = tau
    df["propensity"] = propensity
    return df.round(6)
