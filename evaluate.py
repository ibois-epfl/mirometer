import argparse
import cv2
import os
import mirometer
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def main(target_colour, theoretical_area_mm2=10000):

    images = [file for file in os.listdir("evaluation_data") if file.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG'))]
    print(f"Found {len(images)} images in evaluation_data directory.")
    area_ratios = []
    for file in images:
        print(f"Processing file: {file}")
        corners_per_tag, ids, rejected_corners = mirometer.tag_detection.detect_tags(os.path.join("evaluation_data", file), tag_size=20, tag_type=mirometer.tag_detection.TagType.STAG)
        ppm = 10
        corrected_image, scale = mirometer.image_correction.correct_full_image_using_tag(os.path.join("evaluation_data", file), 
                                                                                         corners_per_tag,
                                                                                         tag_size_mm=20,
                                                                                         space_between_tags_mm=5, 
                                                                                         output_ppmm=ppm)
        mask = mirometer.colour_patch_detection.detect_colour_patch(corrected_image, 
                                                                    target_colour, 
                                                                    hue_tolerance=40,
                                                                    sat_tolerance=60,
                                                                    val_tolerance=100)
        illustration = mirometer.colour_patch_detection.draw_mask_on_image(corrected_image, mask)
        illustration_with_contours, area_contour = mirometer.colour_patch_detection.draw_contour(corrected_image, mask)
        file_name = os.path.splitext(file)[0]
        tagged_image = mirometer.visualisation.add_text_on_mask(illustration_with_contours, 
                                                                mask, 
                                                                f"Area: {area_contour/ppm**2:.2f} mm^2 \n Theoretical Area: {theoretical_area_mm2} mm^2")
        cv2.imwrite(f"evaluation_results/{file_name}_tagged_image.png", tagged_image)
        print(f"Detected area: {area_contour/ppm**2:.2f} mm^2, Theoretical area: {theoretical_area_mm2} mm^2")
        area_ratio = area_contour / (theoretical_area_mm2 * ppm**2)
        area_ratios.append(area_ratio)

    print(f"Average area ratio: {sum(area_ratios) / len(area_ratios):.4f}")

    mirometer.visualisation.plot_evaluation(area_ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirometer CLI")
    parser.add_argument("-c", "--colour", type=int, nargs=3, help="Colour as an (r, g, b) tuple, e.g., (255, 0, 0)")
    parser.add_argument("-a", "--theoretical_area_mm2", type=float, default=10000, help="Theoretical area in mm^2")
    args = parser.parse_args()
    main(args.colour, args.theoretical_area_mm2)
