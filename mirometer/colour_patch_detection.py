import cv2
import numpy as np

def detect_colour_patch(image, target_colour, hue_tolerance=10, sat_tolerance=40, val_tolerance=100):
    """
    Detects a colour patch with better tolerance to lighting variations.

    Uses HSV color space to:
      - Restrict hues (narrow hue_tolerance)
      - Tolerate shadows/overexposure (wide val_tolerance)

    Args:
        image (numpy.ndarray): Input BGR image (OpenCV format).
        target_colour (tuple): Target colour as (r, g, b) tuple.
        hue_tolerance (int): Hue range ±tolerance (0-180, keep small to restrict other hues).
        sat_tolerance (int): Saturation range ±tolerance (0-255).
        val_tolerance (int): Value range ±tolerance (0-255, keep high for shadows/overexposure).

    Returns:
        mask (numpy.ndarray): Binary mask of the detected colour patch.
    """

    target_bgr = np.uint8([[target_colour]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = target_hsv

    lower_hue = max(h - hue_tolerance, 0)
    upper_hue = min(h + hue_tolerance, 180)

    lower_bound = np.array([lower_hue, max(s - sat_tolerance, 0), max(v - val_tolerance, 0)])
    upper_bound = np.array([upper_hue, min(s + sat_tolerance, 255), min(v + val_tolerance, 255)])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return mask


def draw_contour(image, mask):
    """
    Draws contours of the detected colour patch on the original image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask of the detected colour patch.
    
    Returns:
        result_image (numpy.ndarray): Image with contours drawn on it.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")
    contour_areas = [cv2.contourArea(c) for c in contours]
    print(f"Contour areas: {contour_areas}")

    result_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

    return result_image


def draw_mask_on_image(image, mask):
    """
    Draws the detected mask on the original image.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask of the detected colour patch.
    
    Returns:
        result_image (numpy.ndarray): Image with the mask drawn on it.
    """
    coloured_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    coloured_mask[np.where((coloured_mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]  # Red for detected areas

    result_image = cv2.addWeighted(image, 0.7, coloured_mask, 0.3, 0)

    return result_image