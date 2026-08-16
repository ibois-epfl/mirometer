import argparse
import cv2
import mirometer

def main(target_colour, image_path):

    corners_per_tag, ids, rejected_corners = mirometer.tag_detection.detect_tags(image_path, tag_size=20, tag_type=mirometer.tag_detection.TagType.STAG)
    for corner in corners_per_tag:
        print(f"Detected tag corners: {corner}")
    ppm = 10
    corrected_image, scale = mirometer.image_correction.correct_full_image_using_tag(image_path, 
                                                                                     corners_per_tag,
                                                                                     tag_size_mm=20,
                                                                                     space_between_tags_mm=5, 
                                                                                     output_ppmm=ppm)
    mask = mirometer.colour_patch_detection.detect_colour_patch(corrected_image, 
                                                                target_colour, 
                                                                hue_tolerance=20,
                                                                sat_tolerance=50,
                                                                val_tolerance=100)
    illustration = mirometer.colour_patch_detection.draw_mask_on_image(corrected_image, mask)
    illustration_with_contours, area_contour = mirometer.colour_patch_detection.draw_contour(corrected_image, mask)
    tagged_image = mirometer.visualisation.add_text_on_mask(illustration_with_contours, 
                                                            mask, 
                                                            f"Area: {area_contour/ppm**2:.2f} mm^2 \n Theoretical Area: 10000 mm^2")
    # mirometer.visualisation.visualize_image(tagged_image)
    cv2.imwrite("images/tagged_image.png", tagged_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirometer CLI")
    parser.add_argument("-i", "--image_path", type=str, help="Path to the input image")
    parser.add_argument("-c", "--colour", type=int, nargs=3, help="Colour as an (r, g, b) tuple, e.g., (255, 0, 0)")
    args = parser.parse_args()
    main(args.colour, args.image_path)
