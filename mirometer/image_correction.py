import cv2
import numpy as np

def correct_full_image_using_tag(image_path, corners, tag_size_mm=20, output_ppmm=10):
    """
    Rectifies the *entire image* using the tag corners, preserving surroundings.

    Args:
        image_path (str): Path to the input image.
        corners (np.array): 4x2 array of tag corner coordinates (any order).
        tag_size_mm (float): Physical tag size in mm (default: 20mm).
        output_ppmm (float): Pixels per mm for the tag in output (default: 10).

    Returns:
        corrected_image (numpy.ndarray): Full perspective-corrected image.
        scale (float): Pixels per mm (== output_ppmm).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    h, w = image.shape[:2]
    corners = np.array(corners, dtype=np.float32)

    if corners.shape != (4, 2):
        raise ValueError("Corners must be a 4x2 array of (x,y) points.")

    corners = sort_corners(corners)

    tag_size_px = int(tag_size_mm * output_ppmm)
    dst_tag = np.array([
        [0, 0],                     # top-left
        [tag_size_px, 0],           # top-right
        [tag_size_px, tag_size_px], # bottom-right
        [0, tag_size_px]            # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(corners, dst_tag)

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

def sort_corners(corners):
    """
    Sorts 4 corners to [top-left, top-right, bottom-right, bottom-left] order.
    Args:
        corners (np.array): 4x2 array of corner points in image coordinates.

    Returns:
        sorted_corners (np.array): 4x2 array of corners sorted as [top-left, top-right, bottom-right, bottom-left].
    """
    centroid = np.mean(corners, axis=0)

    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)

    sorted_corners = corners[sorted_indices]

    if sorted_corners[0, 1] > sorted_corners[1, 1]:
        sorted_corners = np.roll(sorted_corners, 1, axis=0)

    return sorted_corners