#!/usr/bin/env python3
import sys
import math
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from imageToEquation import YeganehTranscriber, compileLatexToPdf


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe directory of images into Yeganeh equations.")
    parser.add_argument("--inputDir", default="./test_images", type=str, help="Directory containing images.")
    parser.add_argument("--outputDir", default="./transcriptions_batch", type=str, help="Output destination folder.")
    parser.add_argument("--numLayers", default=16, type=int, help="Number of mathematical layers.")
    parser.add_argument("--numIterations", default=200, type=int, help="Iterations per image.")
    parser.add_argument("--compilePdf", action="store_true", help="Compile generated LaTeX to PDF.")
    parser.add_argument("--device", default=None, type=str, help="Device (cuda, mps, cpu).")
    args = parser.parse_args()

    inputDir = Path(args.inputDir)
    outputDir = Path(args.outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    supportedExts = {".png", ".jpg", ".jpeg", ".webp"}
    imageFiles = sorted([f for f in inputDir.iterdir() if f.suffix.lower() in supportedExts])

    if not imageFiles:
        print(f"No supported images found in {inputDir}")
        sys.exit(0)

    print(f"Found {len(imageFiles)} images in {inputDir}. Processing on device: {args.device or 'auto'}")
    transcriber = YeganehTranscriber(deviceStr=args.device)

    summaryList = []

    for fileIdx, imgPath in enumerate(imageFiles, 1):
        cleanTitle = imgPath.stem.replace("_", " ").title()
        print(f"[{fileIdx}/{len(imageFiles)}] Transcribing '{imgPath.name}'...")

        latexCode, renderTensor, artworkSpec = transcriber.transcribe(
            str(imgPath),
            numLayers=args.numLayers,
            numIterations=args.numIterations,
            titleStr=cleanTitle,
            verboseOutput=False,
        )

        baseStem = imgPath.stem
        texPath = outputDir / f"{baseStem}_equations.tex"
        with open(texPath, "w", encoding="utf-8") as fHandle:
            fHandle.write(latexCode)

        renderArray = (renderTensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        renderImg = Image.fromarray(renderArray)
        renderPath = outputDir / f"{baseStem}_render.png"
        renderImg.save(renderPath, "PNG")

        origPil = Image.open(imgPath).convert("RGB")
        targetDisplay = origPil.resize(renderImg.size, Image.Resampling.LANCZOS)
        diffArray = np.abs(np.array(targetDisplay).astype(float) - renderArray.astype(float))
        diffImg = Image.fromarray(np.clip(diffArray * 2.5, 0, 255).astype("uint8"))

        compImg = Image.new("RGB", (renderImg.width * 3, renderImg.height))
        compImg.paste(targetDisplay, (0, 0))
        compImg.paste(renderImg, (renderImg.width, 0))
        compImg.paste(diffImg, (renderImg.width * 2, 0))
        compPath = outputDir / f"{baseStem}_comparison.png"
        compImg.save(compPath, "PNG")

        mseVal = np.mean(diffArray ** 2)
        psnrVal = 10 * math.log10(255 ** 2 / max(mseVal, 1e-8))

        pdfPath = compileLatexToPdf(texPath) if args.compilePdf else None

        summaryList.append({
            "name": imgPath.name,
            "psnr": psnrVal,
            "layers": len(artworkSpec.layers),
            "tex": texPath.name,
            "pdf": pdfPath.name if pdfPath else None,
            "comp": compPath.name,
        })
        print(f"  [+] Complete: {psnrVal:.2f} dB PSNR | Layers: {len(artworkSpec.layers)}")

    # Write summary catalog
    catalogLines = [
        "# Batch Transcription Catalog",
        "",
        f"Processed {len(summaryList)} images.",
        "",
        "| # | Image | PSNR (dB) | Layers | LaTeX | PDF | Comparison |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, row in enumerate(summaryList, 1):
        pdfLink = f"[PDF]({row['pdf']})" if row['pdf'] else "-"
        catalogLines.append(
            f"| {idx} | **{row['name']}** | {row['psnr']:.2f} | {row['layers']} | [LaTeX]({row['tex']}) | {pdfLink} | ![{row['name']}]({row['comp']}) |"
        )

    with open(outputDir / "CATALOG.md", "w", encoding="utf-8") as fHandle:
        fHandle.write("\n".join(catalogLines) + "\n")
    print(f"Batch catalog written to {outputDir / 'CATALOG.md'}")


if __name__ == "__main__":
    main()
