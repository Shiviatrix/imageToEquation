#!/usr/bin/env python3
"""
generate_fresh_test_comparisons.py
Generates fresh, scale-invariant full-canvas comparisons for test images.
Exports rendered equations, 3-panel comparison images, clean LaTeX, and compiled PDFs.
"""

import sys
import json
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np

sys.path.insert(0, "src")
from imageToEquation.hpcEngine import HpcConvergenceEngine, compileLatexToPdf
from imageToEquation.differentiableYeganeh import DifferentiableYeganehArtwork

def main():
    test_dir = Path("/Users/babayaga/.cache/kagglehub/datasets/ezzzio/random-images/versions/1/dataset/test")
    out_dir = Path("test_comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Curated set of test images
    targets = [
        ("test_image1013", test_dir / "image1013.jpg", 150),
        ("test_image1014", test_dir / "image1014.jpg", 150),
        ("test_image100", test_dir / "image100.jpg", 150),
        ("test_image10", test_dir / "image10.jpg", 150),
        ("test_image1019", test_dir / "image1019.jpg", 150),
    ]

    engine = HpcConvergenceEngine(
        targetSsim=0.85,
        targetPsnr=28.0,
        targetLoss=0.03,
        maxLayers=36,
        patienceSteps=20,
        outputDir=str(out_dir)
    )

    results = []
    print("=== Generating Fresh Test Comparisons ===")
    for art_name, img_path, steps in targets:
        if not img_path.exists():
            print(f"Skipping {img_path} (not found)")
            continue

        print(f"\n---> Transcribing {art_name} ({img_path.name}) for {steps} steps...")
        t0 = time.time()
        res = engine.transcribeImageUntilConverged(
            imagePath=str(img_path),
            artworkName=art_name,
            maxTotalSteps=steps
        )
        elapsed = time.time() - t0

        res["elapsedSec"] = elapsed
        results.append(res)
        print(f"Done {art_name}: SSIM={res['ssim']:.4f}, PSNR={res['psnr']:.2f}dB, Layers={res['layers']}, Time={elapsed:.1f}s")

    # Save summary metadata
    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAll fresh test comparisons generated successfully!")

if __name__ == "__main__":
    main()
