import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imutils

def resize_and_maintain_aspect_ratio(input_image, new_width):
    orig_width, orig_height = input_image.shape[1], input_image.shape[0]
    ratio = new_width / float(orig_width)
    new_height = int(orig_height * ratio)
    dim = (new_width, new_height)
    return cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)

def apply_grayscale_blur_and_threshold(img, method="mean", blocksize=91, c=7):
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C if method == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    thresh = cv2.adaptiveThreshold(gray, maxValue=255,
                                   adaptiveMethod=adaptiveMethod,
                                   thresholdType=cv2.THRESH_BINARY,
                                   blockSize=blocksize, C=c)
    return cv2.bitwise_not(thresh)

def get_quadrilateral_points_in_order(approx_arr):
    if approx_arr.shape == (4, 1, 2):
        approx_arr = np.squeeze(approx_arr, axis=1)

    max_x = int(1.1 * np.max(approx_arr[:, 0]))
    origin_1, origin_2 = [0, 0], [max_x, 0]

    distances_1 = [np.linalg.norm(point - origin_1) for point in approx_arr]
    distances_2 = [np.linalg.norm(point - origin_2) for point in approx_arr]

    tl_idx, br_idx = np.argmin(distances_1), np.argmax(distances_1)

    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = np.inf
    dist_arr[br_idx] = np.inf
    tr_idx = np.argmin(dist_arr)

    dist_arr = distances_2.copy()
    dist_arr[tl_idx] = -np.inf
    dist_arr[br_idx] = -np.inf
    bl_idx = np.argmax(dist_arr)

    return np.array([approx_arr[tl_idx], approx_arr[tr_idx], approx_arr[br_idx], approx_arr[bl_idx]])

def perform_four_point_transform(input_img, src_corners, pad=10):
    src_corners = get_quadrilateral_points_in_order(src_corners).astype('float32')
    tl, tr, br, bl = src_corners

    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(bottom_width), int(top_width))

    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(left_height), int(right_height))

    dest_img_corners = np.array([[0+pad, 0+pad],
                                 [max_width-1-pad, 0+pad],
                                 [max_width-1-pad, max_height-1-pad],
                                 [0+pad, max_height-1-pad]], dtype='float32')

    M = cv2.getPerspectiveTransform(src=src_corners, dst=dest_img_corners)
    warped_img = cv2.warpPerspective(input_img, M, (max_width, max_height))
    return M, warped_img

def center_and_resize_digit(cell_img):
    contours, _ = cv2.findContours(cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(cell_img, (28, 28), interpolation=cv2.INTER_AREA)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    digit = cell_img[y:y+h, x:x+w]

    max_side = max(w, h)
    if max_side == 0:
        return cv2.resize(cell_img, (28, 28))
        
    scale = 20.0 / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    
    if new_w <= 0 or new_h <= 0:
        return cv2.resize(cell_img, (28, 28))

    resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    start_y, start_x = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit
    return canvas

def check_for_digit_in_cell_image(img, area_threshold=5, apply_border=False):
    cell_img = img.copy()
    if apply_border:
        border_fraction = 0.07
        y_border_px, x_border_px = int(border_fraction * cell_img.shape[0]), int(border_fraction * cell_img.shape[1])
        cell_img[:, 0:x_border_px] = 0
        cell_img[:, -x_border_px:] = 0
        cell_img[0:y_border_px, :] = 0
        cell_img[-y_border_px:, :] = 0

    contours = cv2.findContours(cell_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour_area = cv2.contourArea(contours[0])
        contour_percentage_area = 100 * largest_contour_area / (cell_img.shape[0] * cell_img.shape[1])
        return contour_percentage_area > area_threshold, cell_img
    
    return False, cell_img

def locate_cells_within_grid(grid_img):
    valid_cells = []
    grid_area = grid_img.shape[0] * grid_img.shape[1]
    grid_img_thresh = apply_grayscale_blur_and_threshold(grid_img, method="mean", blocksize=91, c=7)

    contours = cv2.findContours(grid_img_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if contours:
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            contour_fractional_area = cv2.contourArea(contour) / grid_area

            if len(approx) == 4 and 0.005 < contour_fractional_area < 0.015:
                mask = np.zeros_like(grid_img_thresh)
                cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
                
                y_px, x_px = np.where(mask==255)
                cell_image = grid_img_thresh[min(y_px):max(y_px)+1, min(x_px):max(x_px)+1]
                
                digit_is_present, cell_image = check_for_digit_in_cell_image(cell_image, area_threshold=5, apply_border=True)
                cell_image = cv2.erode(cell_image, np.ones((3, 3), np.uint8), iterations=1)
                
                cell_image = center_and_resize_digit(cell_image) if digit_is_present else np.zeros((28, 28), dtype=np.uint8)
                
                moments = cv2.moments(contour)
                valid_cells.append({
                    'img': cell_image,
                    'contains_digit': digit_is_present,
                    'x_centroid': int(moments['m10'] / moments['m00']),
                    'y_centroid': int(moments['m01'] / moments['m00'])
                })
    return valid_cells

def sort_cells_into_grid(cells):
    max_x, max_y = max(c['x_centroid'] for c in cells), max(c['y_centroid'] for c in cells)
    cell_w, cell_h = (max_x * 1.1) / 9.0, (max_y * 1.1) / 9.0

    for cell in cells:
        cell['grid_row'] = min(int(cell['y_centroid'] / cell_h), 8)
        cell['grid_col'] = min(int(cell['x_centroid'] / cell_w), 8)

    return sorted(cells, key=lambda c: (c['grid_row'], c['grid_col']))

def find_grid_contour_candidates(img):
    M_matrices, warped_images, contour_grid_candidates = [], [], []
    img_area = img.shape[0] * img.shape[1]
    thresh = apply_grayscale_blur_and_threshold(img, blocksize=41, c=8)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sorted(imutils.grab_contours(contours), key=cv2.contourArea, reverse=True)
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            
            if len(approx) == 4 and (cv2.contourArea(contour) / img_area) > 0.1:
                approx = get_quadrilateral_points_in_order(approx)
                M, warped_img = perform_four_point_transform(img, approx, pad=30)
                M_matrices.append(M)
                warped_images.append(warped_img)
                contour_grid_candidates.append(contour)

    return (M_matrices, warped_images, contour_grid_candidates) if warped_images else (None, None, None)

def get_valid_cells_from_image(img):
    M_matrices, warped_images, _ = find_grid_contour_candidates(img)
    if not warped_images: raise Exception("No grid candidates found")

    for i, grid_image in enumerate(warped_images):
        valid_cells = locate_cells_within_grid(grid_image)
        if len(valid_cells) == 81:
            return sort_cells_into_grid(valid_cells), M_matrices[i], grid_image

    raise Exception("Unable to find required number of cells (81)")

def get_predicted_sudoku_grid_torch(model, cells, device):
    digit_images = np.array([cell['img'] for cell in cells if cell['contains_digit']])
    
    if len(digit_images) == 0: return np.zeros((9, 9), dtype=int)

    tensor_images = torch.from_numpy(digit_images).float().unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(tensor_images)
        pred_labels = torch.argmax(outputs, dim=1).cpu().numpy() + 1

    indices = np.where([cell['contains_digit'] for cell in cells])[0]
    grid_array = np.zeros(81, dtype=int)
    grid_array[indices] = pred_labels
    return np.reshape(grid_array, (9, 9))

def plot_cell_images_in_grid(cells):
    """
    Plot all cells in 9x9 grid layout.
    """
    width, height = 9*28, 9*28
    main_img = np.zeros((height, width))

    for i, cell in enumerate(cells):
        row, col = np.array(divmod(i, 9))
        cell_image = cells[i]['img'].copy()
        main_img[row*28:(row+1)*28, col*28:(col+1)*28] = cell_image

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(main_img, cmap='gray')
    ax.set_title('Extracted and Sorted 81 Cells (9×9 Grid Layout)', fontweight='bold')
    ax.axis('off')
    return fig

def generate_solution_image(full_image, board_image, cells_list, solved_board_arr, M_matrix):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = board_image.shape[:2]
    solution_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    flattened_board_array = np.array(solved_board_arr).flatten()
    
    for i, cell in enumerate(cells_list):
        if not cell['contains_digit']:
            text = str(flattened_board_array[i])
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = int((cell['x_centroid'] - textsize[0] / 2))
            text_y = int((cell['y_centroid'] + textsize[1] / 2))
            cv2.putText(solution_img, text, (text_x, text_y), font, 1.3, (0, 0, 0), 2)
    
    unwarped_img = cv2.warpPerspective(
        solution_img, M_matrix, (full_image.shape[1], full_image.shape[0]),
        flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )
    
    annotated = full_image.copy()
    mask = np.all(unwarped_img < 50, axis=-1)
    annotated[mask] = (255, 0, 0) # Red color for digits
    return annotated