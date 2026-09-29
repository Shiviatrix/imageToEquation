#!/usr/bin/env python3
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from imageToEquation import HpcConvergenceEngine, OnlineImageStream, downloadUrl


def updateCatalogFile(catalogItems, outDir: Path):
    catalogMd = outDir / "HPC_CATALOG.md"
    summaryJson = outDir / "hpc_catalog.json"

    with open(summaryJson, "w") as fHandle:
        json.dump(catalogItems, fHandle, indent=2)

    totalCount = len(catalogItems)
    avgPsnr = sum(item["psnr"] for item in catalogItems) / max(totalCount, 1)

    linesList = [
        "# Continuous Transcription Catalog",
        "",
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Artworks transcribed: {totalCount}",
        f"Average PSNR: {avgPsnr:.2f} dB",
        "",
        "| # | Artwork | SSIM | PSNR | Loss | Layers | Steps | Time | LaTeX | PDF | Comparison |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, item in enumerate(catalogItems, 1):
        nameStr = item.get("name", "untitled")
        ssimVal = item.get("ssim", 0.0)
        ssimStr = f"{ssimVal:.4f}" if ssimVal > 0 else "-"
        psnrVal = item.get("psnr", 0.0)
        psnrStr = f"{psnrVal:.2f} dB"
        lossVal = item.get("finalLoss", item.get("final_loss", 0.0))
        lossStr = f"{lossVal:.4f}"
        layersVal = item.get("layers", 0)
        stepsVal = item.get("convergedSteps", item.get("converged_steps", 0))
        timeVal = f"{item.get('elapsedSeconds', item.get('elapsed_seconds', 0))}s"
        texPath = item.get("latexPath", item.get("latex_path", ""))
        texName = Path(texPath).name if texPath else "-"
        pdfPath = item.get("pdfPath", item.get("pdf_path", ""))
        pdfName = Path(pdfPath).name if pdfPath else "-"
        compPath = item.get("comparisonPath", item.get("comparison_path", ""))
        compName = Path(compPath).name if compPath else ""
        pdfLink = f"[PDF]({pdfName})" if pdfPath else "-"
        compImg = f"![{nameStr}]({compName})" if compName else "-"

        linesList.append(f"| {idx} | **{nameStr}** | {ssimStr} | {psnrStr} | {lossStr} | {layersVal} | {stepsVal} | {timeVal} | [LaTeX]({texName}) | {pdfLink} | {compImg} |")

    with open(catalogMd, "w", encoding="utf-8") as fHandle:
        fHandle.write("\n".join(linesList) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Continuous HPC image-to-equation worker.")
    parser.add_argument("--mode", default="curated", choices=["curated", "wikimedia", "picsum", "mixed", "url", "local", "kaggle"], help="Image source feed.")
    parser.add_argument("--dataset", default="ezzzio/random-images", type=str, help="Kaggle dataset handle.")
    parser.add_argument("--split", default="both", choices=["train", "test", "both"], help="Kaggle split to process.")
    parser.add_argument("--url", default=None, type=str, help="Direct image URL.")
    parser.add_argument("--input", default=None, type=str, help="Local file or directory.")
    parser.add_argument("--query", default="mathematical art", type=str, help="Search query for Wikimedia.")
    parser.add_argument("--targetSsim", default=0.82, type=float, help="Target SSIM structural score.")
    parser.add_argument("--targetPsnr", default=26.0, type=float, help="Target PSNR to reach.")
    parser.add_argument("--targetLoss", default=0.04, type=float, help="Target loss threshold.")
    parser.add_argument("--maxLayers", default=128, type=int, help="Maximum adaptive layers.")
    parser.add_argument("--patienceSteps", default=25, type=int, help="Patience before adaptive layer insertion.")
    parser.add_argument("--maxImages", default=0, type=int, help="Number of images to process (0 for infinite).")
    parser.add_argument("--device", default=None, type=str, help="Device (cuda, mps, cpu).")
    parser.add_argument("--outputDir", default="hpc_transcriptions", type=str, help="Output destination folder.")
    args = parser.parse_args()

    outFolder = Path(args.outputDir)
    outFolder.mkdir(parents=True, exist_ok=True)

    hpcEngine = HpcConvergenceEngine(
        targetSsim=args.targetSsim,
        targetPsnr=args.targetPsnr,
        targetLoss=args.targetLoss,
        maxLayers=args.maxLayers,
        patienceSteps=args.patienceSteps,
        deviceStr=args.device,
        outputDir=str(outFolder)
    )

    catalogJson = outFolder / "hpc_catalog.json"
    catalogList = []
    if catalogJson.exists():
        try:
            with open(catalogJson, "r") as fHandle:
                catalogList = json.load(fHandle)
        except Exception:
            catalogList = []

    processedNames = {item["name"] for item in catalogList if "name" in item}

    if args.mode == "kaggle":
        try:
            import kagglehub
            datasetPath = kagglehub.dataset_download(args.dataset)
            print("Path to dataset files:", datasetPath, flush=True)
            datasetDir = Path(datasetPath) / "dataset"
        except Exception as err:
            print(f"Error accessing Kaggle dataset {args.dataset}: {err}", flush=True)
            sys.exit(1)

        trainFiles = sorted(list((datasetDir / "train").glob("*.jpg")) + list((datasetDir / "train").glob("*.png"))) if (datasetDir / "train").exists() else []
        testFiles = sorted(list((datasetDir / "test").glob("*.jpg")) + list((datasetDir / "test").glob("*.png"))) if (datasetDir / "test").exists() else []

        if args.split == "train":
            fileQueue = trainFiles
        elif args.split == "test":
            fileQueue = testFiles
        else:
            fileQueue = []
            maxCount = max(len(trainFiles), len(testFiles))
            for idx in range(maxCount):
                if idx < len(trainFiles):
                    fileQueue.append(trainFiles[idx])
                if idx < len(testFiles):
                    fileQueue.append(testFiles[idx])

        print(f"Loaded Kaggle dataset ({len(trainFiles)} train, {len(testFiles)} test). Queued {len(fileQueue)} images.", flush=True)

        for qIdx, imgFile in enumerate(fileQueue, 1):
            if args.maxImages > 0 and qIdx > args.maxImages:
                print(f"Reached maxImages limit ({args.maxImages}).", flush=True)
                break
            artworkName = f"{imgFile.parent.name}_{imgFile.stem}"
            if artworkName in processedNames:
                continue

            print(f"\n================ [{qIdx}/{len(fileQueue)}] Transcribing {artworkName} ({imgFile.name}) ================", flush=True)
            try:
                metaData = hpcEngine.transcribeImageUntilConverged(str(imgFile), artworkName=artworkName)
                catalogList.append(metaData)
                processedNames.add(artworkName)
                updateCatalogFile(catalogList, outFolder)
                print(f"[Done] {artworkName} | PSNR: {metaData['psnr']:.2f}dB | Loss: {metaData['finalLoss']:.5f} | Layers: {metaData['layers']}", flush=True)
            except Exception as err:
                print(f"[Error] Failed transcribing {artworkName}: {err}", flush=True)
        return

    if args.mode == "url":
        if not args.url:
            print("Error: --url is required when mode is 'url'.")
            sys.exit(1)
        destPath = outFolder / "downloaded_target.png"
        downloaded = downloadUrl(args.url, destPath)
        metaData = hpcEngine.transcribeImageUntilConverged(str(downloaded), artworkName="online_target")
        catalogList.append(metaData)
        updateCatalogFile(catalogList, outFolder)
        return

    if args.mode == "local":
        inputLoc = Path(args.input)
        if inputLoc.is_file():
            metaData = hpcEngine.transcribeImageUntilConverged(str(inputLoc))
            catalogList.append(metaData)
            updateCatalogFile(catalogList, outFolder)
        elif inputLoc.is_dir():
            files = sorted([f for f in inputLoc.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
            print(f"Found {len(files)} images in {inputLoc}. Beginning continuous transcription...", flush=True)
            for fIdx, fItem in enumerate(files, 1):
                if fItem.stem in processedNames:
                    continue
                if args.maxImages > 0 and fIdx > args.maxImages:
                    break
                print(f"\n================ [{fIdx}/{len(files)}] Processing {fItem.name} ================", flush=True)
                try:
                    metaData = hpcEngine.transcribeImageUntilConverged(str(fItem), artworkName=fItem.stem)
                    catalogList.append(metaData)
                    processedNames.add(fItem.stem)
                    updateCatalogFile(catalogList, outFolder)
                except Exception as err:
                    print(f"Error processing {fItem.name}: {err}", flush=True)
        return

    streamObj = OnlineImageStream(streamMode=args.mode, searchQuery=args.query, cacheDir=str(outFolder / "downloads"))
    imageCounter = 0

    while True:
        if args.maxImages > 0 and imageCounter >= args.maxImages:
            print(f"Reached limit of {args.maxImages} images.")
            break

        imageCounter += 1
        itemData = streamObj.nextImage()
        imgLoc = itemData["path"]
        imgName = itemData["name"]

        print(f"\n[Worker] Processing #{imageCounter}: {imgName} ({itemData['title']})")
        try:
            metaData = hpcEngine.transcribeImageUntilConverged(str(imgLoc), artworkName=imgName)
            catalogList.append(metaData)
            updateCatalogFile(catalogList, outFolder)
            print(f"[Worker] Finished: {imgName} (PSNR: {metaData['psnr']:.2f} dB, Steps: {metaData['convergedSteps']})")
        except Exception as err:
            print(f"[Worker] Error processing {imgName}: {err}")

        time.sleep(2)


if __name__ == "__main__":
    main()
