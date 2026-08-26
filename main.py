import argparse
import json
import os
from pathlib import Path
import cv2
import mirometer

def get_image_paths(path):
    """Get list of image paths from a path (file, directory, or glob pattern)."""
    import glob
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.PNG', '.JPEG', '.BMP'}

    # Check if it's a directory FIRST
    if os.path.isdir(path):
        return [
            str(f) for f in Path(path).iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

    # Then check for glob patterns
    matched = glob.glob(path)
    if matched:
        return [p for p in matched if Path(p).suffix.lower() in valid_extensions]

    # Single file
    if os.path.isfile(path) and Path(path).suffix.lower() in valid_extensions:
        return [path]

    raise FileNotFoundError(f"No valid image files found at path: {path}")

def main(target_colour, input_path, tag_size, space_between_tags, hue_tolerance, sat_tolerance, val_tolerance):
    image_paths = get_image_paths(input_path)
    print(f"Found {len(image_paths)} image(s) at {input_path}")
    results_dic = {}
    for image_path in image_paths:
        corners_per_tag, ids, rejected_corners = mirometer.tag_detection.detect_tags(
            image_path, tag_size=tag_size, tag_type=mirometer.tag_detection.TagType.STAG)
        ppm = 10
        corrected_image, scale = mirometer.image_correction.correct_full_image_using_tag(
            image_path,
            corners_per_tag,
            tag_size_mm=tag_size,
            space_between_tags_mm=space_between_tags,
            output_ppmm=ppm)
        mask = mirometer.colour_patch_detection.detect_colour_patch(
            corrected_image,
            target_colour,
            hue_tolerance=hue_tolerance,
            sat_tolerance=sat_tolerance,
            val_tolerance=val_tolerance)
        illustration = mirometer.colour_patch_detection.draw_mask_on_image(corrected_image, mask)
        illustration_with_contours, area_contour = mirometer.colour_patch_detection.draw_contour(corrected_image, mask)
        tagged_image = mirometer.visualisation.add_text_on_mask(
            illustration_with_contours,
            mask,
            f"Area: {int(area_contour/ppm**2)} mm^2")
        results_dic[image_path] = area_contour/ppm**2
        os.makedirs("results", exist_ok=True)
        cv2.imwrite(f"results/tagged_image_{os.path.basename(image_path)}", tagged_image)

    with open("results/results.json", "w") as f:
        json.dump(results_dic, f)
    return results_dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirometer CLI")
    parser.add_argument("-i", "--input_path", type=str, help="Path to the input image, or images directory")
    parser.add_argument("-c", "--colour", type=int, nargs=3, help="Colour as an (r, g, b) tuple, e.g., (255, 0, 0)")
    parser.add_argument("-t", "--tag_size", type=float, default=20, help="Physical tag size in mm (default: 20mm)")
    parser.add_argument("-s", "--space_between_tags", type=float, default=5, help="Physical space between tags in mm (default: 5mm)")
    parser.add_argument("-ht", "--hue_tolerance", type=int, default=20, help="Hue tolerance for colour detection (default: 20)")
    parser.add_argument("-st", "--sat_tolerance", type=int, default=50, help="Saturation tolerance for colour detection (default: 50)")
    parser.add_argument("-vt", "--val_tolerance", type=int, default=50, help="Value tolerance for colour detection (default: 50)")
    args = parser.parse_args()
    main(args.colour, args.input_path, args.tag_size, args.space_between_tags, args.hue_tolerance, args.sat_tolerance, args.val_tolerance)
