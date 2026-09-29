from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .yeganehAst import YeganehLayerSpec, YeganehArtworkSpec, rgbToChannelPolynomial, YeganehTextureTerm


def yeganehContinuousClamp(tensorInput: Tensor) -> Tensor:
    return tensorInput + (torch.clamp(tensorInput, 0.0, 1.0) - tensorInput).detach()


class YeganehGrammarTexture(nn.Module):
    def __init__(self, numTerms: int = 6):
        super().__init__()
        self.numTerms = numTerms
        initCarrier = torch.tensor([
            [16.0, 6.0], [6.0, 20.0], [28.0, 12.0],
            [12.0, 36.0], [48.0, 20.0], [20.0, 56.0]
        ], dtype=torch.float32)
        self.carrierFreq = nn.Parameter(initCarrier)
        self.modFreq = nn.Parameter(torch.tensor([
            [4.0, 2.0], [2.0, 5.0], [6.0, 3.0],
            [3.0, 7.0], [8.0, 4.0], [4.0, 8.0]
        ], dtype=torch.float32))
        self.modDepth = nn.Parameter(torch.zeros(numTerms, dtype=torch.float32))
        self.phases = nn.Parameter(torch.zeros(numTerms, dtype=torch.float32))
        self.amps = nn.Parameter(torch.zeros(numTerms, 3, dtype=torch.float32))
        self.register_buffer("powers", torch.tensor([2, 4, 6, 8, 12, 16], dtype=torch.float32))

    def forward(self, diffX: Tensor, diffY: Tensor, envelope: Tensor, channelV: float) -> Tensor:
        cIdx = max(0, min(2, int(round(channelV))))
        dx = diffX.unsqueeze(0)
        dy = diffY.unsqueeze(0)
        u = self.carrierFreq[:, 0:1].unsqueeze(-1)
        v = self.carrierFreq[:, 1:2].unsqueeze(-1)
        modU = self.modFreq[:, 0:1].unsqueeze(-1)
        modV = self.modFreq[:, 1:2].unsqueeze(-1)
        beta = self.modDepth.view(-1, 1, 1)
        phi = self.phases.view(-1, 1, 1)
        powV = self.powers.view(-1, 1, 1)
        ampsV = self.amps[:, cIdx].view(-1, 1, 1)

        mod = beta * torch.cos(modU * dx + modV * dy + phi)
        psi = u * dx + v * dy + mod
        waves = (torch.cos(psi) ** powV) * envelope.unsqueeze(0)
        return (ampsV * waves).sum(dim=0)


class DifferentiableYeganehLayer(nn.Module):
    def __init__(self, layerSpec: YeganehLayerSpec, isBackground: bool = False):
        super().__init__()
        self.layerName = layerSpec.name
        self.isBackground = isBackground
        self.layerType = layerSpec.layerType

        polyCoeffs = rgbToChannelPolynomial(*layerSpec.colorRgb)
        self.colorPoly = nn.Parameter(torch.tensor(polyCoeffs, dtype=torch.float32))

        gradX, gradY = getattr(layerSpec, "spatialGrad", (0.0, 0.0))
        self.spatialGrad = nn.Parameter(torch.tensor([gradX, gradY], dtype=torch.float32))

        if not isBackground:
            self.paramCx = nn.Parameter(torch.tensor(float(layerSpec.cx), dtype=torch.float32))
            self.paramCy = nn.Parameter(torch.tensor(float(layerSpec.cy), dtype=torch.float32))
            self.paramRx = nn.Parameter(torch.tensor(float(max(layerSpec.rx, 0.01)), dtype=torch.float32))
            self.paramRy = nn.Parameter(torch.tensor(float(max(layerSpec.ry, 0.01)), dtype=torch.float32))
            self.paramPower = nn.Parameter(torch.tensor(float(layerSpec.power), dtype=torch.float32))
            self.paramSharpness = nn.Parameter(torch.tensor(float(layerSpec.sharpness), dtype=torch.float32))

            harmCount = max(len(layerSpec.harmonicAmplitudes), 8)
            ampsList = list(layerSpec.harmonicAmplitudes) + [0.0] * (harmCount - len(layerSpec.harmonicAmplitudes))
            phasesList = list(layerSpec.harmonicPhases) + [0.0] * (harmCount - len(layerSpec.harmonicPhases))

            self.harmAmps = nn.Parameter(torch.tensor(ampsList[:harmCount], dtype=torch.float32))
            self.harmPhases = nn.Parameter(torch.tensor(phasesList[:harmCount], dtype=torch.float32))
            self.harmOrders = torch.arange(1, harmCount + 1, dtype=torch.float32)

            self.textureGen = YeganehGrammarTexture(numTerms=6)

    def evaluateColor(self, channelV: float, coordX: Tensor, coordY: Tensor) -> Tensor:
        coeffA0 = self.colorPoly[0]
        coeffA1 = self.colorPoly[1]
        coeffA2 = self.colorPoly[2]
        basePoly = coeffA0 + coeffA1 * channelV + coeffA2 * (channelV ** 2)

        if not self.isBackground and hasattr(self, "paramCx") and hasattr(self, "paramCy"):
            diffX = coordX - self.paramCx
            diffY = coordY - self.paramCy
            safeRx = torch.abs(self.paramRx) + 1e-4
            safeRy = torch.abs(self.paramRy) + 1e-4
            r2 = (diffX / safeRx) ** 2 + (diffY / safeRy) ** 2
            envelope = torch.clamp(1.0 - r2, min=0.0) ** 1.5
            texMod = self.textureGen(diffX, diffY, envelope, channelV)
        else:
            diffX = coordX
            diffY = coordY
            texMod = 0.0

        spatialMod = 1.0 + self.spatialGrad[0] * diffX + self.spatialGrad[1] * diffY
        return basePoly * spatialMod + texMod

    def evaluateMask(self, coordX: Tensor, coordY: Tensor) -> Tensor:
        if self.isBackground:
            return torch.ones_like(coordX)

        diffX = coordX - self.paramCx
        diffY = coordY - self.paramCy
        safeRx = torch.abs(self.paramRx) + 1e-4
        safeRy = torch.abs(self.paramRy) + 1e-4

        normX = torch.abs(diffX / safeRx)
        normY = torch.abs(diffY / safeRy)
        safePower = torch.clamp(self.paramPower, 0.5, 4.0)

        thetaAngle = torch.atan2(diffY, diffX + 1e-6)
        harmOrders = self.harmOrders.to(coordX.device).view(-1, 1, 1)
        ampsExpanded = self.harmAmps.view(-1, 1, 1)
        phasesExpanded = self.harmPhases.view(-1, 1, 1)

        harmWaves = ampsExpanded * torch.cos(harmOrders * thetaAngle.unsqueeze(0) + phasesExpanded)
        harmPerturbation = harmWaves.sum(dim=0)

        phiBoundary = 1.0 - normX ** safePower - normY ** safePower + harmPerturbation
        sharpnessVal = torch.clamp(self.paramSharpness, 1.0, 100.0)
        expInner = torch.clamp(-sharpnessVal * phiBoundary, -20.0, 20.0)
        return torch.exp(-torch.exp(expInner))

    def exportSpec(self) -> YeganehLayerSpec:
        polyDetached = self.colorPoly.detach().cpu().numpy()
        colorR = float(polyDetached[0])
        colorG = float(polyDetached[0] + polyDetached[1] + polyDetached[2])
        colorB = float(polyDetached[0] + 2.0 * polyDetached[1] + 4.0 * polyDetached[2])
        gradDetached = self.spatialGrad.detach().cpu().numpy()

        if self.isBackground:
            return YeganehLayerSpec(
                name=self.layerName,
                colorRgb=(colorR, colorG, colorB),
                layerType="background",
                spatialGrad=(float(gradDetached[0]), float(gradDetached[1]))
            )

        texList = []
        if hasattr(self, "textureGen"):
            ampsNp = self.textureGen.amps.detach().cpu().numpy()
            carrierNp = self.textureGen.carrierFreq.detach().cpu().numpy()
            modFreqNp = self.textureGen.modFreq.detach().cpu().numpy()
            modDepthNp = self.textureGen.modDepth.detach().cpu().numpy()
            phasesNp = self.textureGen.phases.detach().cpu().numpy()
            powersList = [int(p) for p in self.textureGen.powers.cpu().numpy()]
            for m in range(self.textureGen.numTerms):
                ampTuple = (float(ampsNp[m, 0]), float(ampsNp[m, 1]), float(ampsNp[m, 2]))
                if max(abs(a) for a in ampTuple) > 0.005:
                    texList.append(YeganehTextureTerm(
                        u=float(carrierNp[m, 0]),
                        v=float(carrierNp[m, 1]),
                        modU=float(modFreqNp[m, 0]),
                        modV=float(modFreqNp[m, 1]),
                        modDepth=float(modDepthNp[m]),
                        phase=float(phasesNp[m]),
                        power=powersList[m],
                        ampRgb=ampTuple,
                    ))

        return YeganehLayerSpec(
            name=self.layerName,
            colorRgb=(colorR, colorG, colorB),
            layerType=self.layerType,
            cx=float(self.paramCx.item()),
            cy=float(self.paramCy.item()),
            rx=float(abs(self.paramRx.item())),
            ry=float(abs(self.paramRy.item())),
            power=float(self.paramPower.item()),
            sharpness=float(self.paramSharpness.item()),
            harmonicAmplitudes=[float(x) for x in self.harmAmps.detach().cpu().numpy()],
            harmonicPhases=[float(x) for x in self.harmPhases.detach().cpu().numpy()],
            spatialGrad=(float(gradDetached[0]), float(gradDetached[1])),
            textures=texList,
        )


class DifferentiableYeganehArtwork(nn.Module):
    def __init__(self, artworkSpec: YeganehArtworkSpec):
        super().__init__()
        self.artworkTitle = artworkSpec.title
        self.originX = nn.Parameter(torch.tensor(float(artworkSpec.x0), dtype=torch.float32))
        self.originY = nn.Parameter(torch.tensor(float(artworkSpec.y0), dtype=torch.float32))
        self.scaleX = nn.Parameter(torch.tensor(float(artworkSpec.scaleX), dtype=torch.float32))
        self.scaleY = nn.Parameter(torch.tensor(float(artworkSpec.scaleY), dtype=torch.float32))

        self.layerList = nn.ModuleList()
        for idx, layerSpec in enumerate(artworkSpec.layers):
            isBg = (idx == 0) or (layerSpec.layerType == "background")
            self.layerList.append(DifferentiableYeganehLayer(layerSpec, isBackground=isBg))

    @property
    def layers(self):
        return self.layerList

    def addLayer(self, layerSpec: YeganehLayerSpec):
        newLayer = DifferentiableYeganehLayer(layerSpec, isBackground=False)
        device = next(self.parameters()).device
        newLayer.to(device)
        self.layerList.append(newLayer)

    def computeCoordinates(self, gridHeight: int, gridWidth: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        rowCoords = torch.linspace(1.0 - 1.0 / gridHeight, -1.0 + 1.0 / gridHeight, gridHeight, device=device)
        colCoords = torch.linspace(-1.0 + 1.0 / gridWidth, 1.0 - 1.0 / gridWidth, gridWidth, device=device)
        gridY, gridX = torch.meshgrid(rowCoords, colCoords, indexing="ij")

        safeSx = torch.abs(self.scaleX) + 0.05
        safeSy = torch.abs(self.scaleY) + 0.05
        coordX = (gridX - self.originX) / safeSx
        coordY = (gridY - self.originY) / safeSy
        return coordX, coordY

    def forward(self, height: int = 256, width: int = 256) -> Tensor:
        device = next(self.parameters()).device
        coordX, coordY = self.computeCoordinates(height, width, device)

        bgLayer = self.layerList[0]
        bgRed = bgLayer.evaluateColor(0.0, coordX, coordY)
        bgGreen = bgLayer.evaluateColor(1.0, coordX, coordY)
        bgBlue = bgLayer.evaluateColor(2.0, coordX, coordY)

        compRed = bgRed.clone()
        compGreen = bgGreen.clone()
        compBlue = bgBlue.clone()

        for layerObj in self.layerList[1:]:
            maskWeight = layerObj.evaluateMask(coordX, coordY)
            layerRed = layerObj.evaluateColor(0.0, coordX, coordY)
            layerGreen = layerObj.evaluateColor(1.0, coordX, coordY)
            layerBlue = layerObj.evaluateColor(2.0, coordX, coordY)

            compRed = (1.0 - maskWeight) * compRed + maskWeight * layerRed
            compGreen = (1.0 - maskWeight) * compGreen + maskWeight * layerGreen
            compBlue = (1.0 - maskWeight) * compBlue + maskWeight * layerBlue

        finalRed = yeganehContinuousClamp(compRed)
        finalGreen = yeganehContinuousClamp(compGreen)
        finalBlue = yeganehContinuousClamp(compBlue)

        return torch.stack([finalRed, finalGreen, finalBlue], dim=0)

    def exportSpec(self) -> YeganehArtworkSpec:
        exportedLayers = [layerObj.exportSpec() for layerObj in self.layerList]
        return YeganehArtworkSpec(
            title=self.artworkTitle,
            x0=float(self.originX.item()),
            y0=float(self.originY.item()),
            scaleX=float(abs(self.scaleX.item())),
            scaleY=float(abs(self.scaleY.item())),
            layers=exportedLayers,
        )
