#!/usr/bin/env python3
"""
Official I2E Decoder CLI

Decompresses a .i2e mathematical format file into a standard image.
Usage:
    i2edec input.i2e output.png
"""

import argparse
import sys
import cv2
from image_to_equation.decoder import decode

def main():
    parser = argparse.ArgumentParser(description="Decode an .i2e format into a standard image.")
    parser.add_argument("input", help="Path to input .i2e file")
    parser.add_argument("output", help="Path to output image (e.g. .jpg, .png)")
    
    args = parser.parse_args()
    
    print(f"Decoding {args.input} -> {args.output}...")
    try:
        rendered_np, stats = decode(args.input)
        # Convert RGB back to BGR for OpenCV save
        out_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, out_bgr)
        print("Success!")
        print(f"Decoded Mode: {stats['mode']}")
        print(f"Dimensions: {stats['width']}x{stats['height']}")
        if stats['source']:
            print(f"Equation Length: {len(stats['source'])} characters")
    except Exception as e:
        print(f"Error decoding .i2e file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
