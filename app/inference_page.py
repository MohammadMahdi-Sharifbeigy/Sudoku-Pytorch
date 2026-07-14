"""Mode 1 — Inference (Solve Sudoku)."""
import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.model import (
    DigitCNN, MultiTaskDigitCNN, UnifiedDigitCNN,
    decode_unified_class, UNIFIED_NUM_CLASSES,
    load_digit_cnn_checkpoint,
)
from src.report_utils import save_inference_report
from src.solver import SudokuSolver
from src.vision import (
    DEFAULT_CELL_THRESHOLD_COMBOS,
    apply_grayscale_blur_and_threshold,
    plot_cell_images_in_grid,
    resize_and_maintain_aspect_ratio,
    sharpen_image,
    get_valid_cells_from_image,
    get_predicted_sudoku_grid_torch,
    generate_solution_image,
)


def vision_get_cells(**kwargs):
    return get_valid_cells_from_image(
        kwargs["img"],
        grid_threshold_combos=kwargs.get("grid_threshold_combos"),
        cell_threshold_combos=kwargs.get("cell_threshold_combos"),
        blur_k=kwargs.get("blur_k", 3),
        area_threshold=kwargs.get("area_threshold", 4.0),
        erode_enabled=kwargs.get("erode_enabled", True),
        contour_erode_kernel_size=kwargs.get("contour_erode_kernel_size", 3),
        contour_erode_iterations=kwargs.get("contour_erode_iterations", 1),
        slice_erode_kernel_size=kwargs.get("slice_erode_kernel_size", 2),
        slice_erode_iterations=kwargs.get("slice_erode_iterations", 3),
        digit_scale=kwargs.get("digit_scale", 20),
        should_stop=kwargs.get("should_stop"),
    )

from app.cancel import RunCancelled, clear_cancel, raise_if_cancelled, render_stop_button
from app.debug_utils import save_debug_outputs
from app.onnx_wrapper import OnnxInferenceSession, DigitOnlyModelWrapper, UnifiedDigitOnlyWrapper
from app.preprocess_config import (
    DEFAULT_GRID_THRESHOLD_COMBOS, PREPROCESS_DEFAULTS,
    applied_preprocess_config, current_preprocess_draft,
    format_threshold_combos, init_preprocess_draft,
    parse_threshold_combos, render_preprocess_sidebar,
    threshold_combo_label,
)


_MODEL_PATHS = {
    'persian_pt':   'models/best_model_persian.pt',
    'english_pt':   'models/best_model_english.pt',
    'default_pt':   'models/best_model.pt',
    'mt_pt':        'models/best_model_multitask.pt',
    'unified_pt':   'models/best_model_unified20.pt',
    'persian_onnx': 'models/best_model_persian.onnx',
    'default_onnx': 'models/best_model.onnx',
    'mt_onnx':      'models/best_model_multitask.onnx',
    'unified_onnx': 'models/best_model_unified20.onnx',
}


def _get_per_cell_predictions(model, cells, device, is_multitask=False, is_unified=False):
    is_onnx_mt = (isinstance(model, OnnxInferenceSession)
                  and model.n_outputs > 1 and is_multitask)
    results = []
    for cell in cells:
        if not cell['contains_digit']:
            entry = {'label': 0, 'confidence': 1.0, 'has_digit': False}
            if is_multitask or is_unified:
                entry['lang_label'] = None
                entry['lang_confidence'] = None
            results.append(entry)
            continue

        img    = cell['img'].astype('float32') / 255.0
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_unified:
                logits20 = model(tensor)
                probs20  = torch.softmax(logits20, dim=1).cpu().numpy()[0]
                cls_pred = int(np.argmax(probs20))
                cls_conf = float(probs20[cls_pred])
                digit, lang_name = decode_unified_class(cls_pred)
                results.append({
                    'label': digit, 'confidence': cls_conf, 'has_digit': True,
                    'lang_label': (0 if lang_name == 'Persian' else 1) if lang_name else None,
                    'lang_confidence': cls_conf,
                })
                continue

            if is_onnx_mt:
                d_out, l_out = model.infer_multitask(tensor)
            elif is_multitask:
                d_out, l_out = model(tensor)
            else:
                d_out = model(tensor)

        if is_multitask or is_onnx_mt:
            d_probs   = torch.softmax(d_out, dim=1).cpu().numpy()[0]
            l_probs   = torch.softmax(l_out, dim=1).cpu().numpy()[0]
            pred      = int(np.argmax(d_probs))
            conf      = float(d_probs[pred])
            lang_pred = int(np.argmax(l_probs))
            lang_conf = float(l_probs[lang_pred])
            results.append({'label': pred, 'confidence': conf, 'has_digit': True,
                             'lang_label': lang_pred, 'lang_confidence': lang_conf})
        else:
            probs = torch.softmax(d_out, dim=1).cpu().numpy()[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            results.append({'label': pred, 'confidence': conf, 'has_digit': True})
    return results


def _render_debug_steps(img, img_original, enable_sharpen, use_nlm, sharpen_center,
                         blur_k, thresh_method, thresh_bs, thresh_c,
                         grid_threshold_combos, selected_grid_combo, selected_grid_combo_label,
                         grid_combo_mode, board_image, pipeline_name, cells,
                         area_thresh, erode_enabled, erode_kernel_size, erode_iterations,
                         slice_erode_kernel_size, slice_erode_iterations):
    with st.expander("Intermediate Processing Steps (debug)", expanded=False):
        st.markdown("#### Step 0 — Sharpening")
        if enable_sharpen:
            _kern = f"[[0,-1,0],[-1,**{sharpen_center}**,-1],[0,-1,0]]"
            st.caption(("NLM → " if use_nlm else "") + f"Laplacian {_kern}")
        else:
            st.caption("Sharpening disabled.")
        c0a, c0b = st.columns(2)
        with c0a:
            st.markdown("**Before**"); st.image(img_original, width='stretch')
        with c0b:
            st.markdown("**After**"); st.image(img, width='stretch')
        st.markdown("---")

        st.markdown("#### Step 1 — Grayscale & Gaussian Blur")
        _gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        c1a, c1b = st.columns(2)
        with c1a:
            st.markdown("**Grayscale**"); st.image(_gray, width='stretch', channels="GRAY")
        with c1b:
            if blur_k > 1:
                _blurred = cv2.GaussianBlur(_gray, (blur_k, blur_k), 0)
                st.markdown(f"**GaussianBlur(ksize={blur_k}×{blur_k})**")
                st.image(_blurred, width='stretch', channels="GRAY")
            else:
                st.markdown("**No blur** (blur_k=1)"); st.image(_gray, width='stretch', channels="GRAY")
        st.markdown("---")

        st.markdown("#### Step 2a — Custom Threshold Preview")
        st.caption(f"method=**{thresh_method}**  blocksize=**{thresh_bs}**  C=**{thresh_c}**")
        _t_custom = apply_grayscale_blur_and_threshold(
            img, method=thresh_method, blocksize=thresh_bs, c=thresh_c, blur_k=blur_k)
        c2a, c2b = st.columns(2)
        with c2a:
            st.image(_t_custom, width='stretch', channels="GRAY")
        with c2b:
            st.caption(f"Selected: `{selected_grid_combo_label}` | Mode: `{grid_combo_mode}`")

        with st.expander("Step 2b — All Threshold Combos (click to expand)", expanded=False):
            for _idx in range(0, len(grid_threshold_combos), 2):
                _cols = st.columns(2)
                for _off, (_col, (_bs, _c)) in enumerate(
                        zip(_cols, grid_threshold_combos[_idx:_idx + 2])):
                    _t = apply_grayscale_blur_and_threshold(img, blocksize=_bs, c=_c, blur_k=blur_k)
                    with _col:
                        _sel  = (_bs, _c) == selected_grid_combo
                        _lbl  = threshold_combo_label(_idx + _off, (_bs, _c))
                        st.markdown(f"**{'Selected - ' if _sel else ''}{_lbl}**")
                        st.image(_t, width='stretch', channels="GRAY")


def _render_cell_grid(cells, per_cell, is_multitask, is_unified):
    with st.expander("Cell-by-Cell Extraction & Prediction Details", expanded=True):
        st.markdown(
            "Each cell shows its **extracted image**, **detected** status, "
            "**predicted digit**, and **confidence**. "
            "Bands: high (≥80%), medium (50–80%), low (<50%)."
        )
        st.markdown("---")
        COLS = 9
        hdr_cols = st.columns(COLS)
        for c in range(COLS):
            with hdr_cols[c]:
                st.markdown(
                    f"<div style='text-align:center;color:#888;font-size:11px;'>Col {c}</div>",
                    unsafe_allow_html=True)

        for row in range(9):
            row_cols = st.columns(COLS)
            for col in range(COLS):
                idx  = row * 9 + col
                cell = cells[idx]
                info = per_cell[idx]
                with row_cols[col]:
                    st.image(cell['img'], width=60, channels="GRAY")
                    if not info['has_digit']:
                        st.markdown(
                            "<div style='text-align:center;font-size:11px;color:#888;'>Empty</div>",
                            unsafe_allow_html=True)
                    else:
                        label    = info['label']
                        conf_pct = info['confidence'] * 100
                        band = "High" if conf_pct >= 80 else ("Medium" if conf_pct >= 50 else "Low")
                        lang_badge = ''
                        if (is_multitask or is_unified) and info.get('lang_label') is not None:
                            ln = 'FA' if info['lang_label'] == 0 else 'EN'
                            lc = info['lang_confidence'] * 100
                            lang_badge = (
                                f"<div style='text-align:center;font-size:9px;color:#888;'>"
                                f"{ln} {lc:.0f}%</div>"
                            )
                        st.markdown(
                            f"<div style='text-align:center;font-size:13px;font-weight:bold;'>{label}</div>"
                            f"<div style='text-align:center;font-size:10px;color:#aaa;'>{conf_pct:.1f}%</div>"
                            f"<div style='text-align:center;font-size:9px;color:#888;'>{band}</div>"
                            f"{lang_badge}",
                            unsafe_allow_html=True)
            if row in (2, 5):
                st.markdown("<hr style='border-color:#444;margin:4px 0;'>", unsafe_allow_html=True)


def render_inference_page(device: torch.device) -> None:
    render_stop_button("inference", "Stop inference")

    # ── Sidebar: model selection ──────────────────────────────────────
    st.sidebar.markdown("### Model Selection")

    use_persian = st.sidebar.toggle(
        "Persian image (فارسی)", value=False,
        disabled=not os.path.exists(_MODEL_PATHS['persian_pt']),
    )

    available_mt = {}
    if os.path.exists(_MODEL_PATHS['mt_pt']):
        available_mt["Multi-Task CNN (digit + language head)"] = 'mt'
    if os.path.exists(_MODEL_PATHS['unified_pt']):
        available_mt["Unified 20-Class (MobileNet/ShuffleNet)"] = 'unified'

    use_multimodel = st.sidebar.toggle(
        "Auto-detect language (multi-model)", value=False,
        disabled=len(available_mt) == 0,
    )

    if use_multimodel and available_mt:
        selected_mt_label = st.sidebar.radio("Multi-model to use:", list(available_mt.keys()), index=0)
        selected_mt_key   = available_mt[selected_mt_label]
        if selected_mt_key == 'unified':
            unified_bb = st.sidebar.selectbox(
                "Backbone (must match training)", ["mobilenet_v3_small", "shufflenet_v2_x0_5"])
        else:
            unified_bb = "mobilenet_v3_small"
    else:
        selected_mt_key = None
        unified_bb      = "mobilenet_v3_small"

    # ── ONNX ──────────────────────────────────────────────────────────
    use_onnx     = False
    onnx_avail   = False
    onnx_path    = ''
    try:
        import onnxruntime  # noqa
        if use_multimodel:
            onnx_path = _MODEL_PATHS['mt_onnx'] if selected_mt_key == 'mt' else _MODEL_PATHS['unified_onnx']
        elif use_persian:
            onnx_path = _MODEL_PATHS['persian_onnx']
        else:
            onnx_path = _MODEL_PATHS['default_onnx']
        onnx_avail = os.path.exists(onnx_path)
    except ImportError:
        st.sidebar.caption("onnxruntime not installed — ONNX unavailable.")

    if onnx_avail:
        use_onnx = st.sidebar.toggle("Use ONNX (faster ~2-5×)", value=True)

    is_multitask = use_multimodel and selected_mt_key == 'mt'
    is_unified   = use_multimodel and selected_mt_key == 'unified'
    is_persian   = use_persian and not use_multimodel

    # ── Manual ONNX model file picker (overrides auto-selection above) ─
    manual_onnx_selected = False
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Manual ONNX Model")
    models_dir = "models"
    found_onnx = []
    if os.path.isdir(models_dir):
        found_onnx = sorted(
            f for f in os.listdir(models_dir) if f.lower().endswith(".onnx")
        )

    manual_options = ["Auto (use selections above)"] + found_onnx
    manual_choice = st.sidebar.selectbox(
        "Pick a specific .onnx file from models/", manual_options, index=0,
        help="Scans the models/ folder for any .onnx file. Selecting one "
             "overrides the toggles above and forces ONNX inference.",
    )

    manual_onnx_selected = manual_choice != "Auto (use selections above)"
    if manual_onnx_selected:
        try:
            import onnxruntime  # noqa
        except ImportError:
            st.sidebar.error("onnxruntime not installed — cannot use ONNX models.")
            manual_onnx_selected = False

    if manual_onnx_selected:
        onnx_path  = os.path.join(models_dir, manual_choice)
        use_onnx   = True
        onnx_avail = True

        # Guess model type from filename, but let the user override it.
        _name_lower = manual_choice.lower()
        if "unified" in _name_lower:
            _guess = "Unified 20-class"
        elif "multitask" in _name_lower or "multi_task" in _name_lower or "mt" in _name_lower:
            _guess = "Multi-task (digit + language)"
        else:
            _guess = "Standard (digit only)"

        type_options = ["Standard (digit only)", "Multi-task (digit + language)", "Unified 20-class"]
        model_type = st.sidebar.radio(
            "Model type for selected file", type_options,
            index=type_options.index(_guess),
            help="Determines how outputs are decoded. Auto-guessed from the filename.",
        )

        is_multitask = model_type == "Multi-task (digit + language)"
        is_unified   = model_type == "Unified 20-class"
        is_persian   = False

        if is_unified:
            unified_bb = st.sidebar.selectbox(
                "Backbone (must match training)",
                ["mobilenet_v3_small", "shufflenet_v2_x0_5"],
                key="manual_unified_bb",
            )

        st.sidebar.caption(f"Selected file: `{onnx_path}`")

    # ── Sidebar: preprocessing ────────────────────────────────────────
    st.sidebar.markdown("---")
    active_preprocess, preprocess_dirty = render_preprocess_sidebar()

    # Unpack active config
    grid_threshold_combos = parse_threshold_combos(
        active_preprocess["grid_combo_text"], DEFAULT_GRID_THRESHOLD_COMBOS)
    grid_combo_labels     = [threshold_combo_label(i, c) for i, c in enumerate(grid_threshold_combos)]
    sel_label             = active_preprocess["selected_grid_combo_label"]
    if sel_label not in grid_combo_labels:
        sel_label = grid_combo_labels[0]
    sel_idx               = grid_combo_labels.index(sel_label)
    selected_grid_combo   = grid_threshold_combos[sel_idx]
    grid_combo_mode       = active_preprocess["grid_combo_mode"]
    active_grid_combos    = (
        [selected_grid_combo] if grid_combo_mode == "Use selected combo only"
        else grid_threshold_combos
    )
    enable_sharpen          = active_preprocess["enable_sharpen"]
    use_nlm                 = active_preprocess["use_nlm"]
    sharpen_center          = active_preprocess["sharpen_center"]
    blur_k                  = active_preprocess["blur_k"]
    thresh_method           = active_preprocess["thresh_method"]
    thresh_bs               = active_preprocess["thresh_bs"]
    thresh_c                = active_preprocess["thresh_c"]
    area_thresh             = active_preprocess["area_thresh"]
    erode_enabled           = active_preprocess["erode_enabled"]
    erode_kernel_size       = active_preprocess["erode_kernel_size"]
    erode_iterations        = active_preprocess["erode_iterations"]
    slice_erode_kernel_size = active_preprocess["slice_erode_kernel_size"]
    slice_erode_iterations  = active_preprocess["slice_erode_iterations"]
    digit_scale             = active_preprocess.get("digit_scale", 20)

    # ── Main: tabs ────────────────────────────────────────────────────
    tab_solve, tab_debug = st.tabs(["Solve Sudoku", "Preprocessing Debug"])

    # TAB 1 — Solve
    with tab_solve:
        uploaded_file = st.file_uploader(
            "Drag and drop your Sudoku image here", type=["jpg", "png", "jpeg"],
            key="solve_upload")
        if uploaded_file is not None and preprocess_dirty:
            st.info("Preprocessing changes pending — press Apply before processing.")
            uploaded_file = None

        if uploaded_file is not None:
            image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_and_maintain_aspect_ratio(input_image=img, new_width=1000)

            # Load model
            if manual_onnx_selected:
                model        = OnnxInferenceSession(onnx_path)
                vision_model = model
                _lbl = ("multi-task" if is_multitask else
                        "unified-20" if is_unified else "standard")
                st.sidebar.success(f"ONNX (manual) · {os.path.basename(onnx_path)} · {_lbl}")
            elif use_onnx and onnx_avail:
                model        = OnnxInferenceSession(onnx_path)
                vision_model = model
                _lbl = ("multi-task" if is_multitask else
                        "unified-20" if is_unified else
                        "Persian" if is_persian else "English")
                st.sidebar.success(f"ONNX · {_lbl}")
            elif is_unified:
                _pt = UnifiedDigitCNN(backbone=unified_bb, pretrained=False,
                                      num_classes=UNIFIED_NUM_CLASSES).to(device)
                _pt.load_state_dict(torch.load(_MODEL_PATHS['unified_pt'], map_location=device))
                _pt.eval()
                model        = _pt
                vision_model = UnifiedDigitOnlyWrapper(_pt)
                st.sidebar.info(f"PyTorch · unified-20 · {unified_bb}")
            elif is_multitask:
                _pt = MultiTaskDigitCNN(num_digit_classes=10, num_lang_classes=2).to(device)
                _pt.load_state_dict(torch.load(_MODEL_PATHS['mt_pt'], map_location=device))
                _pt.eval()
                model        = _pt
                vision_model = DigitOnlyModelWrapper(_pt)
                st.sidebar.info("PyTorch · multi-task")
            elif is_persian and os.path.exists(_MODEL_PATHS['persian_pt']):
                model, _arch = load_digit_cnn_checkpoint(
                    _MODEL_PATHS['persian_pt'], device, num_classes=10)
                vision_model = model
                _tag = " (legacy arch)" if _arch == 'LegacyDigitCNN' else ""
                st.sidebar.info(f"PyTorch · Persian (Hoda){_tag}")
            else:
                eng_pt = (_MODEL_PATHS['english_pt']
                          if os.path.exists(_MODEL_PATHS['english_pt'])
                          else _MODEL_PATHS['default_pt'])
                if not os.path.exists(eng_pt):
                    st.error("No trained model found. Go to **Model Training** to train one.")
                    st.stop()
                model, _arch = load_digit_cnn_checkpoint(eng_pt, device, num_classes=10)
                vision_model = model
                _tag = " (legacy arch)" if _arch == 'LegacyDigitCNN' else ""
                st.sidebar.info(f"PyTorch · English (default){_tag}")

            img_original = img.copy()
            if enable_sharpen:
                img = sharpen_image(img, use_nlm=use_nlm, center=sharpen_center)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(img_original, width='stretch')

            with st.spinner('Processing image and extracting grid...'):
                try:
                    cells_orig, M_orig, board_orig = None, None, None
                    err_orig = None
                    try:
                        cells_orig, M_orig, board_orig = vision_get_cells(
                            img=img,
                            grid_threshold_combos=active_grid_combos,
                            cell_threshold_combos=DEFAULT_CELL_THRESHOLD_COMBOS,
                            blur_k=blur_k,
                            area_threshold=area_thresh,
                            erode_enabled=erode_enabled,
                            contour_erode_kernel_size=erode_kernel_size,
                            contour_erode_iterations=erode_iterations,
                            slice_erode_kernel_size=slice_erode_kernel_size,
                            slice_erode_iterations=slice_erode_iterations,
                            digit_scale=digit_scale,
                            should_stop=lambda: raise_if_cancelled("inference"),
                        )
                    except RunCancelled:
                        raise
                    except Exception as e:
                        err_orig = str(e)

                    if cells_orig is None:
                        raise Exception(f"Grid extraction failed: {err_orig}")

                    with st.expander("Grid Extraction Preview", expanded=True):
                        count_orig  = sum(c['contains_digit'] for c in cells_orig)
                        cells       = cells_orig
                        M           = M_orig
                        board_image = board_orig
                        pipeline_name = "Original"
                        st.markdown("**Original Pipeline**")
                        gp_left, gp_right = st.columns(2)
                        with gp_left:
                            st.image(board_orig, channels="GRAY",
                                     caption=f"Warped grid ({count_orig} digits detected)")
                        with gp_right:
                            fig_orig = plot_cell_images_in_grid(cells_orig)
                            st.pyplot(fig_orig); plt.close(fig_orig)

                    debug_dir = save_debug_outputs(
                        image_name=uploaded_file.name,
                        img_rgb=img,
                        cells_selected=cells,
                        board_selected=board_image,
                        pipeline_name=pipeline_name,
                        error=err_orig,
                    )
                    st.sidebar.info(f"Debug outputs → `{debug_dir}`")

                    _render_debug_steps(
                        img, img_original, enable_sharpen, use_nlm, sharpen_center,
                        blur_k, thresh_method, thresh_bs, thresh_c,
                        grid_threshold_combos, selected_grid_combo, sel_label,
                        grid_combo_mode, board_image, pipeline_name, cells,
                        area_thresh, erode_enabled, erode_kernel_size, erode_iterations,
                        slice_erode_kernel_size, slice_erode_iterations,
                    )

                    per_cell = _get_per_cell_predictions(
                        model, cells, device, is_multitask=is_multitask, is_unified=is_unified)
                    _render_cell_grid(cells, per_cell, is_multitask, is_unified)

                    grid_array  = get_predicted_sudoku_grid_torch(
                        vision_model, cells, device,
                        should_stop=lambda: raise_if_cancelled("inference"))
                    solver      = SudokuSolver(board=copy.deepcopy(grid_array))
                    solved_board = solver.board if solver.solve() else None

                    os.makedirs('models', exist_ok=True)
                    inf_report_path = save_inference_report(
                        image_name=uploaded_file.name,
                        cells=cells, per_cell_info=per_cell,
                        grid_array=grid_array, solved_board=solved_board,
                        output_path=f'models/reports/inference_report_{uploaded_file.name}.txt',
                    )
                    st.sidebar.success(f"Inference report → `{inf_report_path}`")

                    if solved_board is not None:
                        final_image = generate_solution_image(
                            full_image=img, board_image=board_image,
                            cells_list=cells, solved_board_arr=solved_board, M_matrix=M)
                        with col2:
                            st.markdown("#### Solved Sudoku")
                            st.image(final_image, width='stretch')
                            st.success("Sudoku solved successfully!")
                        st.markdown("### Digital Representation")
                        matrix_df = pd.DataFrame(solved_board)
                        st.dataframe(
                            matrix_df.style.set_properties(
                                **{'text-align': 'center', 'font-weight': 'bold'}),
                            width='stretch')
                        with open(inf_report_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "Download Inference Report (.txt)",
                                f.read(), 'inference_report.txt', 'text/plain')
                    else:
                        st.error("Grid is invalid or unsolvable. Ensure the image is clear.")
                        st.markdown("**Extracted Grid (before solving):**")
                        st.dataframe(pd.DataFrame(grid_array), width='stretch')

                except RunCancelled as e:
                    st.warning(str(e))
                    clear_cancel("inference")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    st.error(f"Computer Vision Pipeline Error: {e}")
                    st.code(traceback.format_exc())

    # TAB 2 — Preprocessing Debug
    with tab_debug:
        st.markdown(
            "Upload an image to inspect every preprocessing step.  \n"
            "**No digit prediction or solving is performed here.**"
        )
        debug_file = st.file_uploader(
            "Drop a Sudoku image to debug", type=["jpg", "png", "jpeg"], key="debug_upload")
        if debug_file is not None and preprocess_dirty:
            st.info("Preprocessing changes pending — press Apply first.")
            debug_file = None

        if debug_file is not None:
            _bytes  = np.asarray(bytearray(debug_file.read()), dtype=np.uint8)
            img_dbg = cv2.imdecode(_bytes, cv2.IMREAD_COLOR)
            img_dbg = cv2.cvtColor(img_dbg, cv2.COLOR_BGR2RGB)
            img_dbg = resize_and_maintain_aspect_ratio(input_image=img_dbg, new_width=1000)
            img_dbg_orig = img_dbg.copy()
            if enable_sharpen:
                img_dbg = sharpen_image(img_dbg, use_nlm=use_nlm, center=sharpen_center)

            dbg_board, dbg_cells, dbg_pipeline = None, None, "debug"
            with st.spinner("Running vision pipeline for debug…"):
                try:
                    dbg_cells, _, dbg_board = vision_get_cells(
                        img=img_dbg,
                        grid_threshold_combos=active_grid_combos,
                        cell_threshold_combos=DEFAULT_CELL_THRESHOLD_COMBOS,
                        blur_k=blur_k,
                        area_threshold=area_thresh,
                        erode_enabled=erode_enabled,
                        contour_erode_kernel_size=erode_kernel_size,
                        contour_erode_iterations=erode_iterations,
                        slice_erode_kernel_size=slice_erode_kernel_size,
                        slice_erode_iterations=slice_erode_iterations,
                        digit_scale=digit_scale,
                    )
                    dbg_pipeline = "debug-vision"
                except Exception as _dbg_err:
                    st.warning(f"Vision pipeline: {_dbg_err}")

            _render_debug_steps(
                img_dbg, img_dbg_orig, enable_sharpen, use_nlm, sharpen_center,
                blur_k, thresh_method, thresh_bs, thresh_c,
                grid_threshold_combos, selected_grid_combo, sel_label,
                grid_combo_mode, dbg_board, dbg_pipeline, dbg_cells,
                area_thresh, erode_enabled, erode_kernel_size, erode_iterations,
                slice_erode_kernel_size, slice_erode_iterations,
            )