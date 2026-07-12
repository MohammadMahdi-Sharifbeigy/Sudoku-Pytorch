"""Cancellation helpers for long-running Streamlit operations."""
import os
from datetime import datetime

import streamlit as st


class RunCancelled(Exception):
    pass


def _flag_path(scope: str) -> str:
    return os.path.join(os.getcwd(), f".cancel_{scope}")


def request_cancel(scope: str) -> None:
    with open(_flag_path(scope), "w", encoding="utf-8") as f:
        f.write(datetime.now().isoformat())


def clear_cancel(scope: str) -> None:
    path = _flag_path(scope)
    if os.path.exists(path):
        os.remove(path)


def raise_if_cancelled(scope: str) -> None:
    if os.path.exists(_flag_path(scope)):
        raise RunCancelled(f"{scope.title()} stopped by user.")


def render_stop_button(scope: str, label: str) -> None:
    if st.sidebar.button(label, type="secondary", width="stretch", key=f"stop_{scope}_button"):
        request_cancel(scope)
        st.sidebar.warning(f"Stop requested for {scope}.")
