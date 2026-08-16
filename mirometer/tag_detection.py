import stag
import cv2

import enum

class TagType(enum.Enum):
    STAG = "stag"
    ARUCO = "aruco"

def detect_tags(image_path, tag_size=20, tag_type=TagType.STAG):
    """
    Detects tags in the given image

    Args:
        tag_size (int): Size of the tag.
        tag_type (TagType): Type of the tag (STAG or ARUCO).
        image_path (str): Path to the input image.

    Returns:
        corners (list): List of corners of detected tags.
        ids (list): List of IDs of detected tags.
        rejected_corners (list): List of rejected corners.
    """

    if tag_type == TagType.STAG:
        image = cv2.imread(image_path)
        libraryHD = 11
        if image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")
        corners_per_tag, ids, rejected_corners = stag.detectMarkers(image, libraryHD)
        ids_and_corners = list(zip(ids.flatten(), corners_per_tag))
        ids_and_corners.sort(key=lambda x: x[0])
        sorted_corners_per_tag = [corners for _, corners in ids_and_corners]
        stag.drawDetectedMarkers(image, corners_per_tag, ids)
        return sorted_corners_per_tag, ids, rejected_corners