#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from imageToEquation import YeganehTranscriber, compileLatexToPdf


def main():
    parser = argparse.ArgumentParser(description="Transcribe raster image into Yeganeh LaTeX equations.")
    parser.add_argument("--input", required=True, type=str, help="Path to input image.")
    parser.add_argument("--outputDir", default="./output", type=str, help="Directory to save equations and renders.")
    parser.add_argument("--numLayers", default=16, type=int, help="Initial number of mathematical layers.")
    parser.add_argument("--numIterations", default=250, type=int, help="Optimization steps.")
    parser.add_argument("--learningRate", default=0.035, type=float, help="Base learning rate.")
    parser.add_argument("--compilePdf", action="store_true", help="Compile LaTeX equations to PDF with pdflatex.")
    parser.add_argument("--device", default=None, type=str, help="Compute device (cuda, mps, cpu).")
    args = parser.parse_args()

    inputPath = Path(args.input)
    if not inputPath.exists():
        print(f"Error: input file '{inputPath}' does not exist.")
        sys.exit(1)

    outFolder = Path(args.outputDir)
    outFolder.mkdir(parents=True, exist_ok=True)

    transcriber = YeganehTranscriber(deviceStr=args.device)
    latexCode, renderTensor, artworkSpec = transcriber.transcribe(
        str(inputPath),
        numLayers=args.numLayers,
        numIterations=args.numIterations,
        baseLr=args.learningRate,
        titleStr=inputPath.stem.replace("_", " ").title(),
        verboseOutput=True,
    )

    baseStem = inputPath.stem
    texFile = outFolder / f"{baseStem}_equations.tex"
    with open(texFile, "w", encoding="utf-8") as fHandle:
        fHandle.write(latexCode)
    print(f"LaTeX equations saved to: {texFile}")

    renderArray = (renderTensor.cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    renderImg = Image.fromarray(renderArray)
    renderFile = outFolder / f"{baseStem}_render.png"
    renderImg.save(renderFile, "PNG")
    print(f"Rendered image saved to: {renderFile}")

    if args.compilePdf:
        pdfFile = compileLatexToPdf(texFile)
        if pdfFile:
            print(f"PDF compiled successfully: {pdfFile}")
        else:
            print("Notice: pdflatex not found or compilation failed.")


if __name__ == "__main__":
    main()
