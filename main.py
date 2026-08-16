import argparse
import cv2
import mirometer

def main(target_colour, image_path, tag_size, space_between_tags, hue_tolerance, sat_tolerance, val_tolerance):

    corners_per_tag, ids, rejected_corners = mirometer.tag_detection.detect_tags(image_path, tag_size=tag_size, tag_type=mirometer.tag_detection.TagType.STAG)
    ppm = 10
    corrected_image, scale = mirometer.image_correction.correct_full_image_using_tag(image_path, 
                                                                                     corners_per_tag,
                                                                                     tag_size_mm=tag_size,
                                                                                     space_between_tags_mm=space_between_tags, 
                                                                                     output_ppmm=ppm)
    mask = mirometer.colour_patch_detection.detect_colour_patch(corrected_image, 
                                                                target_colour, 
                                                                hue_tolerance=hue_tolerance,
                                                                sat_tolerance=sat_tolerance,
                                                                val_tolerance=val_tolerance)
    illustration = mirometer.colour_patch_detection.draw_mask_on_image(corrected_image, mask)
    illustration_with_contours, area_contour = mirometer.colour_patch_detection.draw_contour(corrected_image, mask)
    tagged_image = mirometer.visualisation.add_text_on_mask(illustration_with_contours, 
                                                            mask, 
                                                            f"Area: {int(area_contour/ppm**2)} mm^2")
    # mirometer.visualisation.visualize_image(tagged_image)
    cv2.imwrite("images/tagged_image.png", tagged_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirometer CLI")
    parser.add_argument("-i", "--image_path", type=str, help="Path to the input image")
    parser.add_argument("-c", "--colour", type=int, nargs=3, help="Colour as an (r, g, b) tuple, e.g., (255, 0, 0)")
    parser.add_argument("-t", "--tag_size", type=float, default=20, help="Physical tag size in mm (default: 20mm)")
    parser.add_argument("-s", "--space_between_tags", type=float, default=5, help="Physical space between tags in mm (default: 5mm)")
    parser.add_argument("-ht", "--hue_tolerance", type=int, default=20, help="Hue tolerance for colour detection (default: 20)")
    parser.add_argument("-st", "--sat_tolerance", type=int, default=50, help="Saturation tolerance for colour detection (default: 50)")
    parser.add_argument("-vt", "--val_tolerance", type=int, default=50, help="Value tolerance for colour detection (default: 50)")
    args = parser.parse_args()
    main(args.colour, args.image_path, args.tag_size, args.space_between_tags, args.hue_tolerance, args.sat_tolerance, args.val_tolerance)
