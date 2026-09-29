#!/usr/bin/env python3
"""
Official I2E Encoder CLI

Compresses an image to the .i2e mathematical format.
Usage:
    i2eenc input_image.jpg output.i2e [--model-path checkpoints/image_to_ast.pt]
"""

import argparse
import sys
from pathlib import Path
from image_to_equation.encoder import encode

def main():
    parser = argparse.ArgumentParser(description="Encode an image into .i2e format.")
    parser.add_argument("input", help="Path to input image (e.g. .jpg, .png)")
    parser.add_argument("output", help="Path to output .i2e file")
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/image_to_ast.pt"), help="Path to the Neural-Symbolic checkpoint")
    
    args = parser.parse_args()
    
    print(f"Encoding {args.input} -> {args.output}...")
    try:
        stats = encode(args.input, args.output, args.model_path)
        print("Success!")
        print(f"Mode: {stats['mode']}")
        print(f"Dimensions: {stats['width']}x{stats['height']}")
        ratio = (1.0 - stats['compressed_size'] / stats['original_size']) * 100
        print(f"Compression Ratio: {ratio:.2f}% (Size: {stats['compressed_size']} bytes)")
    except Exception as e:
        print(f"Error encoding image: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
