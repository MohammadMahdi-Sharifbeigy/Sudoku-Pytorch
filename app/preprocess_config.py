"""Preprocessing parameter management for the Streamlit sidebar."""
import pandas as pd
import streamlit as st

from src.vision import DEFAULT_CELL_THRESHOLD_COMBOS, DEFAULT_GRID_THRESHOLD_COMBOS


def format_threshold_combos(combos) -> str:
    return "; ".join(f"{bs},{c}" for bs, c in combos)


def threshold_combo_label(index: int, combo) -> str:
    blocksize, c_val = combo
    return f"Combo {index + 1}: bs={blocksize}, C={c_val}"


def parse_threshold_combos(raw_text: str, fallback) -> list:
    combos = []
    for chunk in raw_text.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            continue
        try:
            blocksize, c_val = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if blocksize >= 3 and blocksize % 2 == 1 and c_val >= 0:
            combos.append((blocksize, c_val))
    return combos or list(fallback)


def threshold_combos_to_frame(combos) -> pd.DataFrame:
    return pd.DataFrame(
        [{"blocksize": int(bs), "C": int(c)} for bs, c in combos],
        columns=["blocksize", "C"],
    )


PREPROCESS_DEFAULTS = {
    "enable_sharpen":          True,
    "use_nlm":                 False,
    "sharpen_center":          5,
    "blur_k":                  3,
    "thresh_method":           "mean",
    "thresh_bs":               41,
    "thresh_c":                8,
    "area_thresh":             4.0,
    "grid_combo_text":         format_threshold_combos(DEFAULT_GRID_THRESHOLD_COMBOS),
    "selected_grid_combo_label": threshold_combo_label(0, DEFAULT_GRID_THRESHOLD_COMBOS[0]),
    "grid_combo_mode":         "Use selected combo only",
    "erode_enabled":           True,
    "erode_kernel_size":       3,
    "erode_iterations":        1,
    "slice_erode_kernel_size": 2,
    "slice_erode_iterations":  3,
    "digit_scale":             20,
}


def applied_preprocess_config() -> dict:
    if "preprocess_applied" not in st.session_state:
        st.session_state.preprocess_applied = dict(PREPROCESS_DEFAULTS)
    return st.session_state.preprocess_applied


def init_preprocess_draft(config: dict) -> None:
    defaults = {
        "draft_sh_en":           config["enable_sharpen"],
        "draft_sh_nlm":          config["use_nlm"],
        "draft_sh_ctr":          config["sharpen_center"],
        "draft_blur_k":          config["blur_k"],
        "draft_thr_m":           config["thresh_method"],
        "draft_thr_bs":          config["thresh_bs"],
        "draft_thr_c":           config["thresh_c"],
        "draft_area_thr":        config["area_thresh"],
        "draft_grid_combo_text": config["grid_combo_text"],
        "draft_selected_grid_combo": config["selected_grid_combo_label"],
        "draft_grid_combo_mode": config["grid_combo_mode"],
        "draft_erode_enabled":   config["erode_enabled"],
        "draft_erode_kernel":    config["erode_kernel_size"],
        "draft_erode_iter":      config["erode_iterations"],
        "draft_slice_erode_kernel": config["slice_erode_kernel_size"],
        "draft_slice_erode_iter":   config["slice_erode_iterations"],
        "draft_digit_scale":        config["digit_scale"],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_preprocess_draft() -> dict:
    nlm = st.session_state.draft_sh_nlm if st.session_state.draft_sh_en else False
    return {
        "enable_sharpen":          st.session_state.draft_sh_en,
        "use_nlm":                 nlm,
        "sharpen_center":          st.session_state.draft_sh_ctr,
        "blur_k":                  st.session_state.draft_blur_k,
        "thresh_method":           st.session_state.draft_thr_m,
        "thresh_bs":               st.session_state.draft_thr_bs,
        "thresh_c":                st.session_state.draft_thr_c,
        "area_thresh":             st.session_state.draft_area_thr,
        "grid_combo_text":         st.session_state.draft_grid_combo_text,
        "selected_grid_combo_label": st.session_state.draft_selected_grid_combo,
        "grid_combo_mode":         st.session_state.draft_grid_combo_mode,
        "erode_enabled":           st.session_state.draft_erode_enabled,
        "erode_kernel_size":       st.session_state.draft_erode_kernel,
        "erode_iterations":        st.session_state.draft_erode_iter,
        "slice_erode_kernel_size": st.session_state.draft_slice_erode_kernel,
        "slice_erode_iterations":  st.session_state.draft_slice_erode_iter,
        "digit_scale":             st.session_state.draft_digit_scale,
    }


def render_preprocess_sidebar() -> dict:
    """Render sidebar preprocessing controls, return resolved active config."""
    active = applied_preprocess_config()
    init_preprocess_draft(active)

    with st.sidebar.expander("Preprocessing Parameters", expanded=True):
        st.markdown("**Sharpening**")
        st.toggle("Enable", key="draft_sh_en",
                  help="Laplacian kernel sharpening before grid detection.")
        if st.session_state.draft_sh_en:
            st.toggle("NLM denoising first (slow)", key="draft_sh_nlm",
                      help="Non-local means removes noise before sharpening.")
            st.slider("Kernel centre value", min_value=3, max_value=13, step=2,
                      key="draft_sh_ctr",
                      help="Kernel [[0,-1,0],[-1,C,-1],[0,-1,0]]. Higher=stronger.")
        st.markdown("---")
        st.markdown("**Adaptive Threshold**")
        st.select_slider("Gaussian blur kernel size", options=[1, 3, 5, 7], key="draft_blur_k")
        st.selectbox("Threshold method", ["mean", "gaussian"], key="draft_thr_m")
        st.slider("blocksize (odd)", min_value=11, max_value=111, step=2, key="draft_thr_bs")
        st.slider("C (subtracted from mean)", min_value=1, max_value=25, key="draft_thr_c")
        st.markdown("---")
        st.markdown("**Cell Digit Detection**")
        st.slider("Min contour area (%)", min_value=0.5, max_value=10.0,
                  step=0.5, key="draft_area_thr",
                  help="Lower catches thin Persian strokes (~3%).")
        st.markdown("---")
        st.markdown("**Grid Detection Combos**")
        st.caption("Edit rows, then Apply to activate.")
        if st.button("Restore default combos", width="stretch", key="restore_grid_combos"):
            st.session_state.draft_grid_combo_text = format_threshold_combos(DEFAULT_GRID_THRESHOLD_COMBOS)
            st.rerun()

        draft_combos = parse_threshold_combos(
            st.session_state.draft_grid_combo_text, DEFAULT_GRID_THRESHOLD_COMBOS)
        draft_labels = [threshold_combo_label(i, c) for i, c in enumerate(draft_combos)]

        pending = st.session_state.pop("pending_grid_combo_selection", None)
        if pending in draft_labels:
            st.session_state.draft_selected_grid_combo = pending
        if st.session_state.draft_selected_grid_combo not in draft_labels:
            st.session_state.draft_selected_grid_combo = draft_labels[0]

        for idx, (combo_bs, combo_c) in enumerate(draft_combos):
            st.markdown(f"**Combo {idx + 1}**")
            c1, c2 = st.columns(2)
            with c1:
                row_bs = st.number_input(
                    f"Combo {idx + 1} blocksize", min_value=3, max_value=301,
                    value=int(combo_bs), step=2,
                    key=f"grid_combo_bs_{idx}_{combo_bs}_{combo_c}")
            with c2:
                row_c = st.number_input(
                    f"Combo {idx + 1} C", min_value=0, max_value=100,
                    value=int(combo_c), step=1,
                    key=f"grid_combo_c_{idx}_{combo_bs}_{combo_c}")
            row_bs = int(row_bs); row_c = int(row_c)
            if row_bs % 2 == 0:
                row_bs += 1
            a1, a2 = st.columns(2)
            with a1:
                if st.button(f"Update combo {idx + 1}", width="stretch",
                             key=f"update_grid_combo_{idx}"):
                    draft_combos[idx] = (row_bs, row_c)
                    st.session_state.draft_grid_combo_text = format_threshold_combos(draft_combos)
                    st.session_state.pending_grid_combo_selection = threshold_combo_label(
                        idx, draft_combos[idx])
                    st.rerun()
            with a2:
                if st.button(f"Remove combo {idx + 1}", width="stretch",
                             key=f"remove_grid_combo_{idx}",
                             disabled=len(draft_combos) <= 1):
                    draft_combos.pop(idx)
                    st.session_state.draft_grid_combo_text = format_threshold_combos(draft_combos)
                    new_idx = min(idx, len(draft_combos) - 1)
                    st.session_state.pending_grid_combo_selection = threshold_combo_label(
                        new_idx, draft_combos[new_idx])
                    st.rerun()

        st.markdown("**Add New Combo**")
        na1, na2 = st.columns(2)
        with na1:
            new_bs = st.number_input("New blocksize", min_value=3, max_value=301,
                                     value=41, step=2, key="new_combo_bs")
        with na2:
            new_c = st.number_input("New C", min_value=0, max_value=100,
                                    value=8, step=1, key="new_combo_c")
        if st.button("Add combo", width="stretch", key="add_combo_btn"):
            new_bs = int(new_bs)
            if new_bs % 2 == 0:
                new_bs += 1
            draft_combos.append((new_bs, int(new_c)))
            st.session_state.draft_grid_combo_text = format_threshold_combos(draft_combos)
            st.rerun()

        st.markdown("---")
        st.markdown("**Grid Loop Mode**")
        _loop_on = st.toggle(
            "Try all threshold combos (grid loop)",
            value=(st.session_state.get("draft_grid_combo_mode", config["grid_combo_mode"])
                   == "Auto loop all combos"),
            key="draft_grid_loop_toggle",
            help=(
                "Off (default): use only the selected combo — fast, predictable. "
                "On: try all combos and pick the best — slower but more robust on unusual images."
            ),
        )
        st.session_state.draft_grid_combo_mode = (
            "Auto loop all combos" if _loop_on else "Use selected combo only"
        )
        st.markdown("---")
        st.markdown("**Morphological Erosion**")
        st.toggle("Enable erosion", key="draft_erode_enabled")
        st.select_slider("Contour-path kernel", options=[1, 2, 3, 4, 5],
                         key="draft_erode_kernel")
        st.slider("Contour-path iterations", min_value=0, max_value=5, key="draft_erode_iter")
        st.select_slider("Slice-fallback kernel", options=[1, 2, 3, 4, 5],
                         key="draft_slice_erode_kernel")
        st.slider("Slice-fallback iterations", min_value=0, max_value=5,
                  key="draft_slice_erode_iter")
        st.markdown("---")
        st.markdown("**Digit Scaling**")
        st.slider(
            "Digit scale (px within 28×28)", min_value=10, max_value=26, step=1,
            key="draft_digit_scale",
            help=(
                "Controls how large the digit is drawn on the 28×28 canvas. "
                "20 = MNIST-like default. "
                "Lower → more padding (safer for Persian strokes, odd digits). "
                "Higher → tighter crop (may clip thin strokes)."
            ),
        )

        apply_col, reset_col = st.columns(2)
        with apply_col:
            apply_btn = st.button("Apply", type="primary", width="stretch")
        with reset_col:
            reset_btn = st.button("Reset", width="stretch")

        draft = current_preprocess_draft()
        dirty = draft != active

        if apply_btn:
            st.session_state.preprocess_applied = dict(draft)
            active = st.session_state.preprocess_applied
            dirty  = False
            st.success("Settings applied.")

        if reset_btn:
            st.session_state.preprocess_applied = dict(PREPROCESS_DEFAULTS)
            for key in [k for k in st.session_state if k.startswith("draft_")]:
                del st.session_state[key]
            st.rerun()

        active = applied_preprocess_config()
        dirty  = current_preprocess_draft() != active
        if dirty:
            st.warning("Draft changes pending. Press Apply.")
        st.caption(
            f"Active: blur `{active['blur_k']}`, "
            f"combo `{active['selected_grid_combo_label']}`, "
            f"mode `{active['grid_combo_mode']}`, "
            f"erosion `{'on' if active['erode_enabled'] else 'off'}`"
        )

    return active, dirty
