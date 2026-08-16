import cv2

def visualize_image(image):
    """
    Visualizes the given image.

    Args:
        image (numpy.ndarray): Image to be visualized.
    """
    if image is None:
        raise ValueError("Invalid image provided.")

    # Display the image
    cv2.imshow("Image Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()