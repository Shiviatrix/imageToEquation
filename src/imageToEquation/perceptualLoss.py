from __future__ import annotations

import math
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def createGaussianWindow(windowSize: int = 11, sigmaVal: float = 1.5, channels: int = 3) -> Tensor:
    coords = torch.arange(windowSize, dtype=torch.float32) - (windowSize - 1) / 2.0
    gauss = torch.exp(-coords**2 / (2.0 * sigmaVal**2))
    gauss = gauss / gauss.sum()
    kernel2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return kernel2d.expand(channels, 1, windowSize, windowSize).contiguous()


class DifferentiableSsimLoss(nn.Module):
    def __init__(self, windowSize: int = 11, channels: int = 3):
        super().__init__()
        self.windowSize = windowSize
        self.channels = channels
        self.register_buffer("window", createGaussianWindow(windowSize, sigmaVal=1.5, channels=channels))

    def forward(self, img1: Tensor, img2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        window = self.window.to(img1.device)
        padding = self.windowSize // 2

        mu1 = F.conv2d(img1, window, padding=padding, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=self.channels)

        mu1Sq = mu1 ** 2
        mu2Sq = mu2 ** 2
        mu1Mu2 = mu1 * mu2

        sigma1Sq = F.conv2d(img1 * img1, window, padding=padding, groups=self.channels) - mu1Sq
        sigma2Sq = F.conv2d(img2 * img2, window, padding=padding, groups=self.channels) - mu2Sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=self.channels) - mu1Mu2

        ssimMap = ((2.0 * mu1Mu2 + c1) * (2.0 * sigma12 + c2)) / ((mu1Sq + mu2Sq + c1) * (sigma1Sq + sigma2Sq + c2))
        ssimVal = torch.clamp(ssimMap.mean(), -1.0, 1.0)
        lossVal = 1.0 - ssimVal
        return lossVal, ssimVal, ssimMap


class MultiScaleLaplacianLoss(nn.Module):
    def __init__(self, pyramidScales: int = 3):
        super().__init__()
        self.pyramidScales = pyramidScales
        kernelLaplace = torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernelLaplace", kernelLaplace)

    def getEdges(self, imgTensor: Tensor) -> Tensor:
        if imgTensor.ndim == 3:
            imgTensor = imgTensor.unsqueeze(0)
        filterKernel = self.kernelLaplace.repeat(3, 1, 1, 1).to(imgTensor.device)
        return F.conv2d(imgTensor, filterKernel, padding=1, groups=3)

    def forward(self, predTensor: Tensor, targetTensor: Tensor) -> Tensor:
        if predTensor.ndim == 3:
            predTensor = predTensor.unsqueeze(0)
        if targetTensor.ndim == 3:
            targetTensor = targetTensor.unsqueeze(0)

        currPred = predTensor
        currTarget = targetTensor
        accumLoss = torch.tensor(0.0, device=predTensor.device)
        filterKernel = self.kernelLaplace.repeat(3, 1, 1, 1).to(predTensor.device)

        for scaleIdx in range(self.pyramidScales):
            predEdges = F.conv2d(currPred, filterKernel, padding=1, groups=3)
            targetEdges = F.conv2d(currTarget, filterKernel, padding=1, groups=3)
            scaleWeight = 0.5 ** scaleIdx
            accumLoss = accumLoss + scaleWeight * F.l1_loss(predEdges, targetEdges)

            if scaleIdx < self.pyramidScales - 1:
                currPred = F.avg_pool2d(currPred, 2)
                currTarget = F.avg_pool2d(currTarget, 2)

        return accumLoss


class YeganehPerceptualLoss(nn.Module):
    def __init__(self, enableVgg: bool = False):
        super().__init__()
        self.ssimModule = DifferentiableSsimLoss(windowSize=11, channels=3)
        self.laplacianModule = MultiScaleLaplacianLoss(pyramidScales=3)
        self.vggModel = None

        if enableVgg:
            try:
                import torchvision.models as models
                vggNet = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
                for param in vggNet.parameters():
                    param.requires_grad = False
                self.vggModel = vggNet
            except Exception:
                self.vggModel = None

    def forward(self, predTensor: Tensor, targetTensor: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if predTensor.ndim == 3:
            predTensor = predTensor.unsqueeze(0)
        if targetTensor.ndim == 3:
            targetTensor = targetTensor.unsqueeze(0)

        lossSsim, ssimVal, _ = self.ssimModule(predTensor, targetTensor)
        lossEdge = self.laplacianModule(predTensor, targetTensor)
        lossL1 = F.l1_loss(predTensor, targetTensor)
        lossMse = F.mse_loss(predTensor, targetTensor)

        lossVgg = torch.tensor(0.0, device=predTensor.device)
        if self.vggModel is not None:
            meanStat = torch.tensor([0.485, 0.456, 0.406], device=predTensor.device).view(1, 3, 1, 1)
            stdStat = torch.tensor([0.229, 0.224, 0.225], device=predTensor.device).view(1, 3, 1, 1)
            featPred = self.vggModel((predTensor - meanStat) / stdStat)
            featTarget = self.vggModel((targetTensor - meanStat) / stdStat)
            lossVgg = F.mse_loss(featPred, featTarget)

        # Structure (SSIM) and high-frequency edge alignment drive the gradient
        totalLoss = (
            0.45 * lossSsim
            + 0.35 * lossEdge
            + 0.15 * lossL1
            + 0.05 * lossMse
            + (0.05 * lossVgg if self.vggModel is not None else 0.0)
        )

        mseItem = float(lossMse.item())
        psnrItem = -10.0 * math.log10(max(mseItem, 1e-8))

        metricsDict = {
            "ssim": float(ssimVal.item()),
            "edge": float(lossEdge.item()),
            "l1": float(lossL1.item()),
            "mse": mseItem,
            "psnr": psnrItem,
            "loss": float(totalLoss.item()),
        }
        return totalLoss, metricsDict
