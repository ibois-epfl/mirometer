import cv2

def add_text_on_mask(image, mask, text):
    """
    Adds a tag to the image at the center of the detected area.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Binary mask of the detected area.
        text (str): Text to be added as a tag.

    Returns:
        result_image (numpy.ndarray): Image with the tag added.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the mask.")

    largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        raise ValueError("Contour area is zero, cannot compute center.")
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    result_image = image.copy()
    cv2.putText(result_image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    return result_image

def visualize_image(image):
    """
    Visualizes the given image.

    Args:
        image (numpy.ndarray): Image to be visualized.
    """
    if image is None:
        raise ValueError("Invalid image provided.")

    cv2.imshow("Image Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()