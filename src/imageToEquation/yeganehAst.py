from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


def toLatexFraction(val: float, maxDenominator: int = 200, tolerance: float = 1e-3) -> str:
    if abs(val) < 1e-6:
        return "0"
    if abs(round(val) - val) < 1e-4:
        return str(int(round(val)))
    fracVal = Fraction(val).limit_denominator(maxDenominator)
    if abs(float(fracVal) - val) <= tolerance:
        if fracVal.denominator == 1:
            return str(fracVal.numerator)
        if fracVal.numerator < 0:
            return f"-\\frac{{{abs(fracVal.numerator)}}}{{{fracVal.denominator}}}"
        return f"\\frac{{{fracVal.numerator}}}{{{fracVal.denominator}}}"
    return f"{val:.3f}"


def formatSignedTerm(coeff: float, bodyStr: str = "", isFirst: bool = False) -> str:
    if abs(coeff) < 1e-5:
        return ""
    absCoeff = abs(coeff)
    fracStr = toLatexFraction(absCoeff)
    if fracStr == "1" and bodyStr:
        coeffPart = ""
    else:
        coeffPart = fracStr

    if coeff < 0:
        sign = "- " if not isFirst else "-"
    else:
        sign = "+ " if not isFirst else ""

    if bodyStr:
        return f"{sign}{coeffPart}{bodyStr}"
    return f"{sign}{coeffPart}"


def formatOffsetCoordinate(varName: str, offsetVal: float) -> str:
    if abs(offsetVal) < 1e-4:
        return varName
    absOffset = abs(offsetVal)
    fracStr = toLatexFraction(absOffset)
    sign = "-" if offsetVal > 0 else "+"
    return f"\\left({varName} {sign} {fracStr}\\right)"


def rgbToChannelPolynomial(redVal: float, greenVal: float, blueVal: float) -> Tuple[float, float, float]:
    coeffA0 = redVal
    coeffA1 = -1.5 * redVal + 2.0 * greenVal - 0.5 * blueVal
    coeffA2 = 0.5 * redVal - 1.0 * greenVal + 0.5 * blueVal
    return coeffA0, coeffA1, coeffA2


def channelPolynomialToLatex(coeffA0: float, coeffA1: float, coeffA2: float, denominator: int = 40) -> str:
    intC0 = round(coeffA0 * denominator)
    intC1 = round(coeffA1 * denominator)
    intC2 = round(coeffA2 * denominator)

    termsList = []
    if intC0 != 0:
        termsList.append(f"{intC0}")
    if intC1 != 0:
        signStr = "+" if intC1 > 0 and termsList else ("-" if intC1 < 0 else "")
        coeffVal = abs(intC1)
        coeffStr = f"{coeffVal}" if coeffVal != 1 else ""
        if intC1 < 0 and not termsList:
            termsList.append(f"-{coeffStr}v")
        else:
            termsList.append(f"{signStr}{coeffStr}v")
    if intC2 != 0:
        signStr = "+" if intC2 > 0 and termsList else ("-" if intC2 < 0 else "")
        coeffVal = abs(intC2)
        coeffStr = f"{coeffVal}" if coeffVal != 1 else ""
        if intC2 < 0 and not termsList:
            termsList.append(f"-{coeffStr}v^2")
        else:
            termsList.append(f"{signStr}{coeffStr}v^2")

    if not termsList:
        return "0"
    numeratorStr = "".join(termsList)
    if denominator == 1:
        return numeratorStr
    return f"\\frac{{{numeratorStr}}}{{{denominator}}}"


@dataclass
class YeganehTextureTerm:
    u: float
    v: float
    modU: float
    modV: float
    modDepth: float
    phase: float
    power: int
    ampRgb: Tuple[float, float, float]


@dataclass
class YeganehLayerSpec:
    name: str
    colorRgb: Tuple[float, float, float]
    layerType: str = "harmonic"
    cx: float = 0.0
    cy: float = 0.0
    rx: float = 0.5
    ry: float = 0.5
    power: float = 1.0
    sharpness: float = 30.0
    harmonicAmplitudes: List[float] = field(default_factory=lambda: [0.0] * 8)
    harmonicPhases: List[float] = field(default_factory=lambda: [0.0] * 8)
    spatialGrad: Tuple[float, float] = (0.0, 0.0)
    textureWeight: float = 0.0
    textures: List[YeganehTextureTerm] = field(default_factory=list)
    horizonPhase: float = 0.0
    horizonFreq: float = 1.0
    spiralK: float = 0.2
    spiralA: float = 0.5
    latticeSpacingX: float = 0.3
    latticeSpacingY: float = 0.3

    def getPolynomial(self) -> Tuple[float, float, float]:
        return rgbToChannelPolynomial(*self.colorRgb)

    def toLatex(self, layerIdx: int, xVar: str = "X", yVar: str = "Y") -> str:
        idxStr = f"{{{layerIdx}}}"
        coeffA0, coeffA1, coeffA2 = self.getPolynomial()
        basePoly = channelPolynomialToLatex(coeffA0, coeffA1, coeffA2)
        gradX, gradY = self.spatialGrad

        hasTranslation = (abs(self.cx) > 1e-4 or abs(self.cy) > 1e-4)
        if hasTranslation:
            xLocal = f"X_{idxStr}"
            yLocal = f"Y_{idxStr}"
            localCoordDef = (
                f"  {xLocal} &= {formatOffsetCoordinate(xVar, self.cx)}, \\quad "
                f"{yLocal} = {formatOffsetCoordinate(yVar, self.cy)} \\\\"
            )
        else:
            xLocal = xVar
            yLocal = yVar
            localCoordDef = ""

        # Spatial Gradient Modulation
        spatialGradTerms = []
        if abs(gradX) > 0.01:
            spatialGradTerms.append(formatSignedTerm(gradX, xLocal))
        if abs(gradY) > 0.01:
            spatialGradTerms.append(formatSignedTerm(gradY, yLocal))

        spatialMod = ""
        if spatialGradTerms:
            spatialMod = f" \\cdot \\left(1 {' '.join(spatialGradTerms)}\\right)"

        alignLines = []
        if localCoordDef:
            alignLines.append(localCoordDef)

        # High-Frequency Modulation / Texture Terms
        hasTextures = False
        if self.textures:
            texTerms = []
            for tIdx, tex in enumerate(self.textures):
                avgAmp = sum(tex.ampRgb) / 3.0
                if abs(avgAmp) > 0.005:
                    uStr = formatSignedTerm(tex.u, xLocal, isFirst=True)
                    vStr = formatSignedTerm(tex.v, yLocal)
                    carrierArg = f"{uStr} {vStr}".strip()

                    modArg = ""
                    if abs(tex.modDepth) > 0.05:
                        modU = formatSignedTerm(tex.modU, xLocal, isFirst=True)
                        modV = formatSignedTerm(tex.modV, yLocal)
                        modInner = f"{modU} {modV}".strip()
                        modArg = f" {formatSignedTerm(tex.modDepth, f'\\cos\\left({modInner}\\right)')}"

                    ampTerm = formatSignedTerm(avgAmp, f"\\cos^{{{tex.power}}}\\left({carrierArg}{modArg}\\right)", isFirst=(len(texTerms) == 0))
                    texTerms.append(ampTerm)

            if texTerms:
                hasTextures = True
                # Break every 2 terms cleanly across aligned lines
                alignLines.append(f"  \\tau_{idxStr}({xVar},{yVar}) &= {texTerms[0]}")
                for i in range(1, len(texTerms), 2):
                    chunk = " ".join(texTerms[i:i+2])
                    alignLines.append(f"    &\\quad {chunk}")

        if self.layerType == "background":
            colorDef = f"  C_0({xVar},{yVar},v) &= {basePoly}{spatialMod}"
            alignLines.append(colorDef)
            return "\n".join(alignLines)

        if hasTextures:
            colorDef = f"  C_{idxStr}({xVar},{yVar},v) &= {basePoly}{spatialMod} + \\tau_{idxStr}({xVar},{yVar})"
        else:
            colorDef = f"  C_{idxStr}({xVar},{yVar},v) &= {basePoly}{spatialMod}" if spatialMod else f"  C_{idxStr}(v) &= {basePoly}"
        alignLines.append(colorDef)

        # Boundary Contour Function Phi_k
        rxStr = toLatexFraction(abs(self.rx))
        ryStr = toLatexFraction(abs(self.ry))
        xNormTerm = f"\\left(\\frac{{{xLocal}}}{{{rxStr}}}\\right)^{{2}}"
        yNormTerm = f"\\left(\\frac{{{yLocal}}}{{{ryStr}}}\\right)^{{2}}"

        harmTerms = []
        for harmIdx, (ampVal, phaseVal) in enumerate(zip(self.harmonicAmplitudes, self.harmonicPhases), 1):
            if abs(ampVal) > 0.003:
                phaseStr = formatSignedTerm(phaseVal)
                cosArg = f"{harmIdx}\\theta {phaseStr}".strip()
                harmTerms.append(formatSignedTerm(ampVal, f"\\cos\\left({cosArg}\\right)", isFirst=False))

        alignLines.append(f"  \\Phi_{idxStr}({xVar},{yVar}) &= 1 - {xNormTerm} - {yNormTerm}")
        if harmTerms:
            for i in range(0, len(harmTerms), 4):
                chunk = " ".join(harmTerms[i:i+4])
                alignLines.append(f"    &\\quad {chunk}")

        # Double-Exponential Mask W_k
        sharpnessStr = toLatexFraction(abs(self.sharpness))
        alignLines.append(f"  W_{idxStr}({xVar},{yVar}) &= \\exp\\left(-\\exp\\left(-{sharpnessStr}\\cdot\\Phi_{idxStr}({xVar},{yVar})\\right)\\right)")

        return "\n".join(alignLines)


@dataclass
class YeganehArtworkSpec:
    title: str = "Mathematical Artwork"
    x0: float = 0.0
    y0: float = 0.0
    scaleX: float = 1.0
    scaleY: float = 1.0
    layers: List[YeganehLayerSpec] = field(default_factory=list)

    def toLatex(self) -> str:
        cleanTitle = self.title.replace("_", " ").title()
        outputLines = [
            f"\\section*{{{cleanTitle}}}",
            "",
            "\\subsection*{Scale-Invariant Coordinate System}",
            "\\[",
            "  \\tilde{X} = 2\\left(\\frac{x}{W}\\right) - 1, \\quad \\tilde{Y} = 1 - 2\\left(\\frac{y}{H}\\right) \\\\[6pt]",
            f"  X = \\frac{{{formatOffsetCoordinate(r'\tilde{X}', self.x0)}}}{{{toLatexFraction(abs(self.scaleX))}}}, \\quad "
            f"Y = \\frac{{{formatOffsetCoordinate(r'\tilde{Y}', self.y0)}}}{{{toLatexFraction(abs(self.scaleY))}}}, \\quad "
            "\\theta = \\arctan\\left(\\frac{Y}{X}\\right)",
            "\\]",
            "",
            "\\subsection*{Layer Definitions & Boundary Manifolds}",
        ]

        for layerIdx, layerObj in enumerate(self.layers):
            layerLatex = layerObj.toLatex(layerIdx)
            outputLines.append(f"% Layer {layerIdx}: {layerObj.name}")
            outputLines.append("\\begin{align*}")
            outputLines.append(layerLatex)
            outputLines.append("\\end{align*}")
            outputLines.append("")

        outputLines.extend([
            "\\subsection*{Master Composition (Recursive Layer Synthesis)}",
            "\\begin{align*}",
            "  H_v^{(0)}(X,Y) &= C_0(X,Y,v) \\\\[4pt]",
        ])

        for layerIdx in range(1, len(self.layers)):
            idxStr = f"{{{layerIdx}}}"
            prevIdxStr = f"{{{layerIdx - 1}}}"
            outputLines.append(
                f"  H_v^{idxStr}(X,Y) &= \\left(1 - W_{idxStr}(X,Y)\\right)\\cdot H_v^{prevIdxStr}(X,Y) "
                f"+ W_{idxStr}(X,Y)\\cdot C_{idxStr}(X,Y,v) \\\\[3pt]"
            )

        lastIdxStr = f"{{{len(self.layers) - 1}}}" if len(self.layers) > 1 else "{0}"
        outputLines.extend([
            f"  H_v(X,Y) &= H_v^{lastIdxStr}(X,Y)",
            "\\end{align*}",
            "",
            "\\subsection*{Final RGB Channel Evaluation}",
            "\\[",
            "  \\text{Color}(x,y) = \\left( F\\left(H_0(X,Y)\\right), \\; F\\left(H_1(X,Y)\\right), \\; F\\left(H_2(X,Y)\\right) \\right)",
            "\\]",
            "where $F(x) = \\left[255\\,e^{-e^{-1000x}}|x|\\,e^{-e^{1000(x-1)}}\\right]$ performs exact differentiable color clamping to $[0, 255]$."
        ])

        return "\n".join(outputLines)

    def toStandaloneDocument(self) -> str:
        bodyText = self.toLatex()
        documentParts = [
            "\\documentclass[10pt]{article}",
            "\\usepackage{amsmath,amssymb,mathtools}",
            "\\usepackage[margin=0.6in,landscape]{geometry}",
            "\\usepackage{microtype}",
            "\\usepackage{enumitem}",
            "\\allowdisplaybreaks",
            "",
            "\\setlength{\\parindent}{0pt}",
            "\\setlength{\\jot}{4pt}",
            "",
            "\\begin{document}",
            "",
            bodyText,
            "",
            "\\end{document}",
        ]
        return "\n".join(documentParts)
