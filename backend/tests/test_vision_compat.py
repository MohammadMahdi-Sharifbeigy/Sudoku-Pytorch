PUBLIC_API = [
    "resize_and_maintain_aspect_ratio", "apply_grayscale_blur_and_threshold",
    "get_quadrilateral_points_in_order", "perform_four_point_transform",
    "center_and_resize_digit", "check_for_digit_in_cell_image",
    "slice_grid_into_cells", "sort_cells_into_grid", "build_grid_from_partial_cells",
    "get_valid_cells_from_image", "get_cells_classical",
    "get_predicted_sudoku_grid_torch", "get_per_cell_predictions",
    "plot_cell_images_in_grid", "generate_solution_image",
    "get_grid_corners_yolo", "get_cells_yolo", "load_yolo_model",
]


def test_public_api_importable():
    import app.core.vision as vision
    missing = [n for n in PUBLIC_API if not hasattr(vision, n)]
    assert not missing, f"missing exports: {missing}"
