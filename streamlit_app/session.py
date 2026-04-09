from __future__ import annotations

import streamlit as st

from .constants import SESSION_DEFAULTS


def initialize_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_training_result() -> None:
    st.session_state["training_result"] = None


def set_dataset(df, source: str, file_hash: str | None = None) -> None:
    st.session_state["raw_dataset"] = df
    st.session_state["dataset_source"] = source
    st.session_state["dataset_file_hash"] = file_hash
    st.session_state["validation_report"] = None
    st.session_state["training_result"] = None


def set_column_selection(selection: dict) -> None:
    st.session_state["column_selection"] = selection


def set_validation_report(report) -> None:
    st.session_state["validation_report"] = report


def set_training_result(result) -> None:
    st.session_state["training_result"] = result
