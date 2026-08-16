import argparse

import mirometer

def main(target_colour, image_path):
    print("Hello from mirometer!")
    print(f"Image path: {image_path}")
    print(f"Colour: {target_colour}")
    mirometer.tag_detection.detect_tags(image_path, tag_size=20, tag_type=mirometer.tag_detection.TagType.STAG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirometer CLI")
    parser.add_argument("-i", "--image_path", type=str, help="Path to the input image")
    parser.add_argument("-c", "--colour", type=float, nargs=3, help="Colour as an (r, g, b) tuple, e.g., (255, 0, 0)")
    args = parser.parse_args()
    main(args.colour, args.image_path)
