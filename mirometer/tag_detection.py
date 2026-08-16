import stag
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
        list: A list of detected tags.
    """
    print(f"Detecting {tag_type.value} tags in image: {image_path} with tag size: {tag_size}")