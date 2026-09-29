from __future__ import annotations

import os
import math
import time
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from sklearn.cluster import MiniBatchKMeans

from .yeganehAst import YeganehLayerSpec, YeganehArtworkSpec
from .differentiableYeganeh import DifferentiableYeganehArtwork
from .perceptualLoss import YeganehPerceptualLoss


def loadImageTensor(imageInput: Union[str, Path, Image.Image, Tensor], targetSize: Optional[Tuple[int, int]] = None) -> Tensor:
    if isinstance(imageInput, Tensor):
        tensorImg = imageInput.detach().float()
        if tensorImg.ndim == 4:
            tensorImg = tensorImg.squeeze(0)
        if tensorImg.max() > 1.0:
            tensorImg = tensorImg / 255.0
        if targetSize is not None:
            tensorImg = torch.nn.functional.interpolate(
                tensorImg.unsqueeze(0), size=targetSize, mode="bilinear", align_corners=False
            ).squeeze(0)
        return tensorImg

    if isinstance(imageInput, (str, Path)):
        pilImage = Image.open(str(imageInput)).convert("RGB")
    elif isinstance(imageInput, Image.Image):
        pilImage = imageInput.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(imageInput)}")

    if targetSize is not None:
        pilImage = pilImage.resize((targetSize[1], targetSize[0]), Image.Resampling.LANCZOS)

    arrImage = np.array(pilImage, dtype=np.float32) / 255.0
    return torch.tensor(arrImage).permute(2, 0, 1)


class YeganehTranscriber:
    def __init__(self, deviceStr: Optional[str] = None, enableVgg: bool = False):
        if deviceStr:
            self.device = torch.device(deviceStr)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.lossModule = YeganehPerceptualLoss(enableVgg=enableVgg).to(self.device)

    def extractInitialSpec(self, targetTensor: Tensor, numLayers: int = 16, titleStr: str = "Artwork") -> YeganehArtworkSpec:
        channelCount, origH, origW = targetTensor.shape
        originX = origW / 2.0
        originY = origH / 2.0
        scaleNorm = max(origW, origH) / 2.0

        imgNp = targetTensor.permute(1, 2, 0).cpu().numpy()
        bgTop = imgNp[0, :, :].mean(axis=0)
        bgBottom = imgNp[-1, :, :].mean(axis=0)
        bgLeft = imgNp[:, 0, :].mean(axis=0)
        bgRight = imgNp[:, -1, :].mean(axis=0)
        bgRgb = tuple(float(x) for x in (bgTop + bgBottom + bgLeft + bgRight) / 4.0)

        layerSpecs = [
            YeganehLayerSpec(
                name="Background",
                colorRgb=bgRgb,
                layerType="background",
                cx=0.0,
                cy=0.0,
                rx=1.5,
                ry=1.5,
                sharpness=1.0,
            )
        ]

        smallH = 128
        smallW = max(16, int(round(origW * smallH / origH)))
        pilImg = Image.fromarray((np.clip(imgNp, 0.0, 1.0) * 255).astype(np.uint8))
        smallNp = np.array(pilImg.resize((smallW, smallH), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        clusterBudget = min(numLayers + 4, 32)
        pixelGrid = smallNp.reshape(-1, 3)
        kmeans = MiniBatchKMeans(n_clusters=clusterBudget, random_state=42, batch_size=1024).fit(pixelGrid)
        clusterLabels = kmeans.labels_.reshape(smallH, smallW)
        clusterCounts = np.bincount(kmeans.labels_)
        sortedIndices = np.argsort(clusterCounts)[::-1]

        for rankIdx, clusterIdx in enumerate(sortedIndices):
            if len(layerSpecs) >= numLayers:
                break
            clusterRgb = tuple(float(x) for x in kmeans.cluster_centers_[clusterIdx])
            distToBg = math.sqrt(sum((clusterRgb[i] - bgRgb[i]) ** 2 for i in range(3)))
            if distToBg < 0.05 and rankIdx == 0:
                continue

            clusterMask = (clusterLabels == clusterIdx)
            if clusterMask.sum() < 4:
                continue

            pixelCoords = np.argwhere(clusterMask)
            meanRow = pixelCoords[:, 0].mean()
            meanCol = pixelCoords[:, 1].mean()
            stdRow = pixelCoords[:, 0].std() + 3.0
            stdCol = pixelCoords[:, 1].std() + 3.0

            coordCx = float((meanCol + 0.5) / smallW * 2.0 - 1.0)
            coordCy = float(1.0 - (meanRow + 0.5) / smallH * 2.0)
            coordRx = float(max(0.08, min(1.0, (stdCol / smallW) * 2.5)))
            coordRy = float(max(0.08, min(1.0, (stdRow / smallH) * 2.5)))

            layerSpecs.append(
                YeganehLayerSpec(
                    name=f"Primitive_{len(layerSpecs)}",
                    colorRgb=clusterRgb,
                    layerType="harmonic",
                    cx=coordCx,
                    cy=coordCy,
                    rx=coordRx,
                    ry=coordRy,
                    power=1.0,
                    sharpness=45.0,
                    harmonicAmplitudes=[0.05, 0.03, 0.02, 0.015, 0.01, 0.005, 0.005, 0.002],
                    harmonicPhases=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                    textureWeight=0.01,
                )
            )

        return YeganehArtworkSpec(
            title=titleStr,
            x0=0.0,
            y0=0.0,
            scaleX=1.0,
            scaleY=1.0,
            layers=layerSpecs,
        )

    def transcribe(
        self,
        imageInput: Union[str, Path, Image.Image],
        numLayers: int = 18,
        numIterations: int = 250,
        baseLr: float = 0.035,
        progressiveStages: bool = True,
        titleStr: str = "Artwork",
        verboseOutput: bool = False,
    ) -> Tuple[str, Tensor, YeganehArtworkSpec]:
        targetTensor = loadImageTensor(imageInput)
        origHeight, origWidth = targetTensor.shape[1], targetTensor.shape[2]

        artworkSpec = self.extractInitialSpec(targetTensor, numLayers=numLayers, titleStr=titleStr)
        diffModel = DifferentiableYeganehArtwork(artworkSpec).to(self.device)

        if progressiveStages:
            pyramidStages = [
                {"res": 128, "steps": int(numIterations * 0.35), "lr": baseLr},
                {"res": 256, "steps": int(numIterations * 0.40), "lr": baseLr * 0.7},
                {"res": 384, "steps": int(numIterations * 0.25), "lr": baseLr * 0.35},
            ]
        else:
            pyramidStages = [{"res": 256, "steps": numIterations, "lr": baseLr}]

        globalStep = 0
        totalSteps = sum(s["steps"] for s in pyramidStages)

        for stageIdx, stageDict in enumerate(pyramidStages):
            stageRes = stageDict["res"]
            stageSteps = stageDict["steps"]
            stageLr = stageDict["lr"]

            fitWidth = int(round(origWidth * stageRes / max(origHeight, origWidth)))
            fitHeight = int(round(origHeight * stageRes / max(origHeight, origWidth)))
            fitTarget = loadImageTensor(imageInput, targetSize=(fitHeight, fitWidth)).to(self.device)

            optParams = [
                {"params": [p for n, p in diffModel.named_parameters() if "spatialGrad" in n or "color" in n], "lr": stageLr * 1.2},
                {"params": [p for n, p in diffModel.named_parameters() if "harm" in n], "lr": stageLr * 0.8},
                {"params": [p for n, p in diffModel.named_parameters() if "param" in n or "scale" in n], "lr": stageLr * 0.6},
            ]
            optimizer = torch.optim.Adam(optParams if optParams[0]["params"] else diffModel.parameters(), lr=stageLr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stageSteps, eta_min=1e-4)

            for stepIdx in range(1, stageSteps + 1):
                globalStep += 1
                optimizer.zero_grad()
                predTensor = diffModel(height=fitHeight, width=fitWidth)

                lossVal, metricsDict = self.lossModule(predTensor, fitTarget)
                lossVal.backward()

                torch.nn.utils.clip_grad_norm_(diffModel.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                if verboseOutput and (stepIdx % 30 == 0 or stepIdx == stageSteps):
                    psnrVal = -10.0 * math.log10(max(metricsDict["mse"], 1e-8))
                    print(f"  Stage {stageIdx+1}/{len(pyramidStages)} | Step {globalStep:3d}/{totalSteps} | Loss: {lossVal.item():.4f} (PSNR: {psnrVal:.2f} dB)")

            # Adaptive layer allocation between pyramid stages
            if stageIdx < len(pyramidStages) - 1 and len(diffModel.layers) < 26:
                with torch.no_grad():
                    currPred = diffModel(height=fitHeight, width=fitWidth)
                    residualMap = torch.mean(torch.abs(fitTarget - currPred), dim=0)
                    maxResidual, maxResidualIdx = torch.max(residualMap.view(-1), dim=0)

                    if float(maxResidual) > 0.08:
                        errRow = int(maxResidualIdx // fitWidth)
                        errCol = int(maxResidualIdx % fitWidth)
                        coordCx = ((float(errCol) + 0.5) * (origWidth / fitWidth) - float(diffModel.originX.item())) / float(diffModel.scaleX.item())
                        coordCy = (float(diffModel.originY.item()) - (float(errRow) + 0.5) * (origHeight / fitHeight)) / float(diffModel.scaleY.item())
                        targetRgb = tuple(float(c) for c in fitTarget[:, errRow, errCol].cpu())

                        newSpec = YeganehLayerSpec(
                            name=f"Adaptive_{len(diffModel.layers)}",
                            colorRgb=targetRgb,
                            cx=coordCx,
                            cy=coordCy,
                            rx=0.15,
                            ry=0.15,
                            power=1.0,
                            sharpness=55.0,
                            harmonicAmplitudes=[0.05, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005, 0.002],
                            harmonicPhases=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                            textureWeight=0.01,
                            layerType="harmonic",
                        )
                        diffModel.addLayer(newSpec)

        with torch.no_grad():
            renderRes = min(max(origWidth, origHeight), 1024)
            finalRender = diffModel(height=renderRes, width=renderRes)

        finalSpec = diffModel.exportSpec()
        latexSource = finalSpec.toStandaloneDocument()
        return latexSource, finalRender, finalSpec
