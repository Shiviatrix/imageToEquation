from __future__ import annotations

import os
import sys
import time
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from PIL import Image

from .yeganehTranscriber import YeganehTranscriber, loadImageTensor
from .yeganehAst import YeganehLayerSpec, YeganehArtworkSpec
from .differentiableYeganeh import DifferentiableYeganehArtwork


def compileLatexToPdf(texPath: Path) -> Optional[Path]:
    if not shutil.which("pdflatex"):
        return None
    try:
        texPath = Path(texPath)
        outDir = texPath.parent
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", texPath.name],
            cwd=str(outDir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        pdfPath = texPath.with_suffix(".pdf")
        if pdfPath.exists():
            return pdfPath
    except Exception:
        pass
    return None


class HpcConvergenceEngine:
    def __init__(
        self,
        targetSsim: float = 0.82,
        targetPsnr: float = 26.0,
        targetLoss: float = 0.04,
        maxLayers: int = 128,
        patienceSteps: int = 30,
        deviceStr: Optional[str] = None,
        outputDir: str = "hpc_transcriptions",
        checkpointDir: str = "hpc_checkpoints"
    ):
        self.targetSsim = targetSsim
        self.targetPsnr = targetPsnr
        self.targetLoss = targetLoss
        self.maxLayers = maxLayers
        self.patienceSteps = patienceSteps
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.checkpointDir = Path(checkpointDir)
        self.checkpointDir.mkdir(parents=True, exist_ok=True)

        self.transcriber = YeganehTranscriber(deviceStr=deviceStr)
        self.device = self.transcriber.device

    def transcribeImageUntilConverged(
        self,
        imagePath: str,
        artworkName: Optional[str] = None,
        maxTotalSteps: int = 1500
    ) -> Dict[str, Any]:
        imagePath = Path(imagePath)
        if not artworkName:
            artworkName = imagePath.stem

        targetTensor = loadImageTensor(str(imagePath))
        origHeight, origWidth = targetTensor.shape[1], targetTensor.shape[2]

        artworkSpec = self.transcriber.extractInitialSpec(targetTensor, numLayers=16, titleStr=artworkName)
        diffModel = DifferentiableYeganehArtwork(artworkSpec).to(self.device)

        maxDim = max(origWidth, origHeight)
        if maxDim <= 128:
            resolutionList = [max(48, maxDim // 2), maxDim]
        elif maxDim <= 256:
            resolutionList = [64, 128, maxDim]
        elif maxDim <= 512:
            resolutionList = [128, 256, maxDim]
        else:
            resolutionList = [128, 256, 512, min(maxDim, 768)]

        currResIdx = 0

        stepCount = 0
        bestLoss = float("inf")
        bestSsim = 0.0
        bestPsnr = 0.0
        stepsWithoutGain = 0
        startTime = time.time()

        def setupStage(targetRes: int):
            fitW = int(round(origWidth * targetRes / max(origHeight, origWidth)))
            fitH = int(round(origHeight * targetRes / max(origHeight, origWidth)))
            stageTarget = loadImageTensor(str(imagePath), targetSize=(fitH, fitW)).to(self.device)
            stageOpt = torch.optim.Adam(diffModel.parameters(), lr=0.035 * (0.8 ** currResIdx))
            stageSched = torch.optim.lr_scheduler.CosineAnnealingLR(stageOpt, T_max=80, eta_min=1e-4)
            return fitW, fitH, stageTarget, stageOpt, stageSched

        currW, currH, currTarget, optimizer, scheduler = setupStage(resolutionList[currResIdx])

        while stepCount < maxTotalSteps:
            optimizer.zero_grad()
            predTensor = diffModel(height=currH, width=currW)
            lossVal, metricsDict = self.transcriber.lossModule(predTensor, currTarget)
            lossVal.backward()

            torch.nn.utils.clip_grad_norm_(diffModel.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            currentLoss = lossVal.item()
            currentSsim = metricsDict.get("ssim", 0.0)
            currentEdge = metricsDict.get("edge", 0.0)
            currentPsnr = metricsDict.get("psnr", 0.0)

            # Monitor progress on both structural similarity (SSIM) and loss
            if currentLoss < bestLoss - 1e-4 or currentSsim > bestSsim + 0.005:
                if currentLoss < bestLoss:
                    bestLoss = currentLoss
                bestSsim = max(bestSsim, currentSsim)
                bestPsnr = max(bestPsnr, currentPsnr)
                stepsWithoutGain = 0
            else:
                stepsWithoutGain += 1

            if stepCount % 25 == 0 or currentSsim >= self.targetSsim:
                print(
                    f"[{artworkName}] Step {stepCount}/{maxTotalSteps} | "
                    f"Res {currW}x{currH} | Layers {len(diffModel.layers)} | "
                    f"Loss: {currentLoss:.4f} | SSIM: {currentSsim:.4f} (Best: {bestSsim:.4f}) | "
                    f"Edge: {currentEdge:.4f} | PSNR: {currentPsnr:.2f}dB",
                    flush=True
                )

            # Stop when structural realism target is achieved
            if (currentSsim >= self.targetSsim and currentPsnr >= self.targetPsnr) or currentLoss <= self.targetLoss:
                print(f"[{artworkName}] Target criteria achieved! SSIM: {currentSsim:.4f}, PSNR: {currentPsnr:.2f}dB", flush=True)
                break

            # Working backwards from deficit: analyze failure regions and synthesize terms
            if stepsWithoutGain >= self.patienceSteps:
                stepsWithoutGain = 0
                if currResIdx < len(resolutionList) - 1:
                    currResIdx += 1
                    currW, currH, currTarget, optimizer, scheduler = setupStage(resolutionList[currResIdx])
                    print(f"[{artworkName}] Advancing to resolution stage {resolutionList[currResIdx]} ({currW}x{currH})", flush=True)
                elif len(diffModel.layers) < self.maxLayers:
                    with torch.no_grad():
                        currPred = diffModel(height=currH, width=currW)

                        # 1. Structural error from SSIM failure map
                        _, _, ssimMap = self.transcriber.lossModule.ssimModule(currPred, currTarget)
                        structErr = 1.0 - ssimMap.squeeze(0).mean(dim=0)

                        # 2. High-frequency edge discrepancy
                        targetEdges = self.transcriber.lossModule.laplacianModule.getEdges(currTarget).squeeze(0).abs().mean(dim=0)
                        predEdges = self.transcriber.lossModule.laplacianModule.getEdges(currPred).squeeze(0).abs().mean(dim=0)
                        edgeErr = torch.abs(targetEdges - predEdges)

                        # Combined deficit map
                        deficitMap = 0.55 * structErr + 0.45 * edgeErr
                        _, maxIdx = torch.max(deficitMap.view(-1), dim=0)

                        errRow = int(maxIdx // currW)
                        errCol = int(maxIdx % currW)

                        normCol = (float(errCol) + 0.5) / currW * 2.0 - 1.0
                        normRow = 1.0 - (float(errRow) + 0.5) / currH * 2.0
                        safeSx = abs(float(diffModel.scaleX.item())) + 0.05
                        safeSy = abs(float(diffModel.scaleY.item())) + 0.05
                        coordCx = (normCol - float(diffModel.originX.item())) / safeSx
                        coordCy = (normRow - float(diffModel.originY.item())) / safeSy
                        coordCx = max(-0.95, min(0.95, coordCx))
                        coordCy = max(-0.95, min(0.95, coordCy))
                        targetRgb = tuple(float(c) for c in currTarget[:, errRow, errCol].cpu())

                        # Analyze local target gradient to deduce edge direction vs texture
                        grayTarget = currTarget.mean(dim=0)
                        gy = float(grayTarget[min(currH - 1, errRow + 1), errCol] - grayTarget[max(0, errRow - 1), errCol])
                        gx = float(grayTarget[errRow, min(currW - 1, errCol + 1)] - grayTarget[errRow, max(0, errCol - 1)])
                        gradMag = math.hypot(gx, gy)
                        edgeAngle = math.atan2(gy, gx)

                        adaptiveScale = max(0.015, min(0.2, 0.15 * (0.90 ** (len(diffModel.layers) - 16))))

                        # Grammar-guided term synthesis:
                        if gradMag > 0.08:
                            # Sharp boundary detected: Directional stroke + high-power carrier wave
                            layerRx = adaptiveScale * 2.5
                            layerRy = adaptiveScale * 0.7
                            layerPower = 2.0
                            layerSharpness = 90.0
                            normalAngle = edgeAngle
                            uFreq = 32.0 * math.cos(normalAngle)
                            vFreq = 32.0 * math.sin(normalAngle)
                            termPowers = [8, 12, 16, 16, 24, 24]
                            layerTypeStr = "edge_stroke"
                        else:
                            # Regional texture or smooth contour
                            layerRx = adaptiveScale
                            layerRy = adaptiveScale
                            layerPower = 1.0
                            layerSharpness = 50.0
                            uFreq = 16.0
                            vFreq = 16.0
                            termPowers = [2, 4, 6, 8, 12, 16]
                            layerTypeStr = "texture_harmonic"

                        newLayerSpec = YeganehLayerSpec(
                            name=f"Layer_{len(diffModel.layers)}_{layerTypeStr}",
                            colorRgb=targetRgb,
                            cx=coordCx,
                            cy=coordCy,
                            rx=layerRx,
                            ry=layerRy,
                            power=layerPower,
                            sharpness=layerSharpness,
                            harmonicAmplitudes=[0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.002],
                            harmonicPhases=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                            textureWeight=0.02,
                            layerType="harmonic",
                        )
                        diffModel.addLayer(newLayerSpec)

                        # Set carrier frequency in new layer texture module
                        newLayerModule = diffModel.layerList[-1]
                        if hasattr(newLayerModule, "textureGen"):
                            with torch.no_grad():
                                newLayerModule.textureGen.carrierFreq[0, 0] = uFreq
                                newLayerModule.textureGen.carrierFreq[0, 1] = vFreq
                                newLayerModule.textureGen.powers.copy_(torch.tensor(termPowers, dtype=torch.float32))

                        optimizer = torch.optim.Adam(diffModel.parameters(), lr=0.03)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-4)
                        print(
                            f"[{artworkName}] Working backwards: Added {layerTypeStr} {len(diffModel.layers)} at "
                            f"({coordCx:.2f}, {coordCy:.2f}) | GradMag: {gradMag:.3f} | Angle: {math.degrees(edgeAngle):.1f}°",
                            flush=True
                        )
                else:
                    if stepsWithoutGain > self.patienceSteps * 2:
                        break

            stepCount += 1

        totalElapsed = time.time() - startTime
        print(
            f"[{artworkName}] Completed in {stepCount} steps ({totalElapsed:.1f}s) | "
            f"Final Loss: {bestLoss:.5f} | SSIM: {bestSsim:.4f} | PSNR: {bestPsnr:.2f}dB | Layers: {len(diffModel.layers)}",
            flush=True
        )

        # Aspect-Ratio Preserving Final Rendering
        renderRes = min(max(origWidth, origHeight, 512), 1024)
        renderH = int(round(origHeight * renderRes / max(origHeight, origWidth)))
        renderW = int(round(origWidth * renderRes / max(origHeight, origWidth)))

        with torch.no_grad():
            renderArray = diffModel(height=renderH, width=renderW).cpu().numpy()
            renderArray = np.clip(renderArray * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        finalRenderImg = Image.fromarray(renderArray)
        renderPath = self.outputDir / f"{artworkName}_rendered.png"
        finalRenderImg.save(renderPath, "PNG")

        origPil = Image.open(imagePath).convert("RGB")
        targetDisplay = origPil.resize((renderW, renderH), Image.Resampling.LANCZOS)
        diffArray = np.abs(np.array(targetDisplay).astype(np.float32) - renderArray.astype(np.float32))
        diffImg = Image.fromarray(np.clip(diffArray * 2.5, 0, 255).astype(np.uint8))

        compImg = Image.new("RGB", (renderW * 3, renderH))
        compImg.paste(targetDisplay, (0, 0))
        compImg.paste(finalRenderImg, (renderW, 0))
        compImg.paste(diffImg, (renderW * 2, 0))
        compPath = self.outputDir / f"{artworkName}_comparison.png"
        compImg.save(compPath, "PNG")

        exportedSpec = diffModel.exportSpec()
        cleanTitle = artworkName.replace("_", " ").title()
        exportedSpec.title = f"Artwork: {cleanTitle}"
        texPath = self.outputDir / f"{artworkName}_equations.tex"
        texCode = exportedSpec.toStandaloneDocument()
        with open(texPath, "w", encoding="utf-8") as fHandle:
            fHandle.write(texCode)

        pdfPath = compileLatexToPdf(texPath)

        metaDict = {
            "name": artworkName,
            "sourceImage": str(imagePath),
            "ssim": round(float(bestSsim), 4),
            "psnr": round(float(bestPsnr), 2),
            "finalLoss": round(float(bestLoss), 5),
            "layers": int(len(diffModel.layers)),
            "convergedSteps": int(stepCount),
            "elapsedSeconds": round(totalElapsed, 2),
            "renderPath": str(renderPath),
            "comparisonPath": str(compPath),
            "latexPath": str(texPath),
            "pdfPath": str(pdfPath) if pdfPath else None
        }

        metaJsonPath = self.outputDir / f"{artworkName}_meta.json"
        with open(metaJsonPath, "w") as fHandle:
            json.dump(metaDict, fHandle, indent=2)

        return metaDict
