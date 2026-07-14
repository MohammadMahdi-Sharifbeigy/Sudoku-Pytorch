"""Cancellation helpers. Uses threading.Event for instant, thread-safe stop."""
import threading
from datetime import datetime

import streamlit as st


class RunCancelled(Exception):
    pass


# Per-scope stop events (live as long as the process)
_STOP_EVENTS: dict[str, threading.Event] = {}


def get_stop_event(scope: str) -> threading.Event:
    if scope not in _STOP_EVENTS:
        _STOP_EVENTS[scope] = threading.Event()
    return _STOP_EVENTS[scope]


def request_cancel(scope: str) -> None:
    get_stop_event(scope).set()


def clear_cancel(scope: str) -> None:
    get_stop_event(scope).clear()


def raise_if_cancelled(scope: str) -> None:
    if get_stop_event(scope).is_set():
        raise RunCancelled(f"{scope.title()} stopped by user.")


def is_cancelled(scope: str) -> bool:
    return get_stop_event(scope).is_set()


def render_stop_button(scope: str, label: str) -> None:
    if st.sidebar.button(label, type="secondary", use_container_width=True, key=f"stop_{scope}_button"):
        request_cancel(scope)
        st.sidebar.warning("Stop requested.")
