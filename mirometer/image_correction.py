import cv2
import numpy as np

def correct_full_image_using_tag(image_path, corners_per_tag, tag_size_mm=20, space_between_tags_mm=5, output_ppmm=10):
    """
    Rectifies the *entire image* using the tag corners, preserving surroundings.

    Args:
        image_path (str): Path to the input image.
        corners ([np.array]): list of 4x2 arrays of tags corner coordinates (any order).
        tag_size_mm (float): Physical tag size in mm (default: 20mm).
        space_between_tags_mm (float): Physical space between tags in mm (default: 5mm).
        output_ppmm (float): Pixels per mm for the tag in output (default: 10).

    Returns:
        corrected_image (numpy.ndarray): Full perspective-corrected image.
        scale (float): Pixels per mm (== output_ppmm).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    h, w = image.shape[:2]
    
    n_tags = len(corners_per_tag)
    overall_top_left, overall_top_right, overall_bottom_right, overall_bottom_left = None, None, None, None

    for i in range(len(corners_per_tag)):
        tag_corners = corners_per_tag[i][0]
        if tag_corners.shape != (4, 2):
            raise ValueError("Corners must be a 4x2 array of (x,y) points.")
        top_left, top_right, bottom_right, bottom_left = tag_corners[0], tag_corners[1], tag_corners[2], tag_corners[3]
        if i == 0:
            overall_top_left = top_left
            overall_bottom_left = bottom_left
        if i == len(corners_per_tag) - 1:
            overall_top_right = top_right
            overall_bottom_right = bottom_right

    tag_size_px = int(tag_size_mm * output_ppmm)
    dst_tag = np.array([
        [0, 0],                     # top-left
        [n_tags * tag_size_px + (n_tags - 1) * space_between_tags_mm * output_ppmm, 0],           # top-right
        [n_tags * tag_size_px + (n_tags - 1) * space_between_tags_mm * output_ppmm, tag_size_px], # bottom-right
        [0, tag_size_px]            # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(
    np.float32([
        overall_top_left.tolist(),
        overall_top_right.tolist(),
        overall_bottom_right.tolist(),
        overall_bottom_left.tolist()
    ]),
    dst_tag
    )

    src_image_corners = np.array([
        [0, 0], [w, 0], [w, h], [0, h]
    ], dtype=np.float32)
    dst_image_corners = cv2.perspectiveTransform(
        src_image_corners[None, :, :], H
    )[0]

    min_x, min_y = np.min(dst_image_corners, axis=0)
    max_x, max_y = np.max(dst_image_corners, axis=0)

    H_adjusted = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ]) @ H

    output_w = int(np.ceil(max_x - min_x))
    output_h = int(np.ceil(max_y - min_y))

    corrected_image = cv2.warpPerspective(
        image, H_adjusted, (output_w, output_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    return corrected_image, output_ppmm
