from __future__ import annotations

import os
import json
import time
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Any
from PIL import Image
import numpy as np

CURATED_ONLINE_ARTWORKS = [
    {
        "title": "Hamid Naderi Yeganeh - A Bird in Flight",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e5/A_Bird_in_Flight.png",
    },
    {
        "title": "Hamid Naderi Yeganeh - Boat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Boat_by_Hamid_Naderi_Yeganeh.png",
    },
    {
        "title": "Hamid Naderi Yeganeh - Flying Bird",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/87/Flying_Bird.png",
    },
    {
        "title": "Mandelbrot Fractal Detail",
        "url": "https://upload.wikimedia.org/wikipedia/commons/2/21/Mandel_zoom_00_mandelbrot_set.jpg",
    },
    {
        "title": "Julia Set Fractal",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Julia_set_%28golden_ratio_variant%29.png",
    },
    {
        "title": "Harmonograph Mathematical Curve",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/69/Harmonograph_1.png",
    },
    {
        "title": "Islamic Geometric Pattern",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/ce/Girih_tiles_pattern.png",
    },
    {
        "title": "Lorenz Attractor Dynamical System",
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Lorenz_attractor_yb.svg",
    }
]

USER_AGENT = "ImageToEquation/1.0 (https://github.com/Shiviatrix/image-to-equation) Python-urllib"


def downloadUrl(imgUrl: str, destPath: Path, maxRetries: int = 3) -> Path:
    destPath.parent.mkdir(parents=True, exist_ok=True)
    reqObj = urllib.request.Request(imgUrl, headers={"User-Agent": USER_AGENT})

    for attemptIdx in range(maxRetries):
        try:
            with urllib.request.urlopen(reqObj, timeout=25) as respHandle:
                dataBytes = respHandle.read()
            with open(destPath, "wb") as fHandle:
                fHandle.write(dataBytes)

            pilImg = Image.open(destPath).convert("RGB")
            destPng = destPath.with_suffix(".png")
            pilImg.save(destPng, "PNG")
            if destPng != destPath:
                destPath.unlink(missing_ok=True)
            return destPng
        except Exception:
            time.sleep(1.5 * (attemptIdx + 1))

    # Fallback to procedural geometric pattern if network fails
    destPng = destPath.with_suffix(".png")
    gridW, gridH = 800, 800
    gridY, gridX = np.mgrid[:gridH, :gridW]
    radiusVal = np.sqrt((gridX - gridW / 2) ** 2 + (gridY - gridH / 2) ** 2)
    thetaVal = np.arctan2(gridY - gridH / 2, gridX - gridW / 2)
    patternMask = np.sin(0.04 * radiusVal + 5 * thetaVal) > 0

    arrRgb = np.zeros((gridH, gridW, 3), dtype=np.uint8)
    arrRgb[patternMask] = [230, 190, 80]
    arrRgb[~patternMask] = [30, 45, 90]
    Image.fromarray(arrRgb).save(destPng, "PNG")
    return destPng


def fetchWikimediaImages(searchQuery: str = "mathematical art", maxCount: int = 15) -> List[Dict[str, str]]:
    queryArgs = {
        "action": "query",
        "generator": "search",
        "gsrnamespace": "6",
        "gsrsearch": f"{searchQuery} filetype:bitmap",
        "gsrlimit": str(maxCount),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "format": "json"
    }
    apiUrl = f"https://commons.wikimedia.org/w/api.php?{urllib.parse.urlencode(queryArgs)}"
    reqObj = urllib.request.Request(apiUrl, headers={"User-Agent": USER_AGENT})

    matchedImages = []
    try:
        with urllib.request.urlopen(reqObj, timeout=20) as respHandle:
            resJson = json.loads(respHandle.read().decode("utf-8"))

        pagesDict = resJson.get("query", {}).get("pages", {})
        for _, pageData in pagesDict.items():
            infoList = pageData.get("imageinfo", [])
            if not infoList:
                continue
            itemUrl = infoList[0].get("url")
            itemMime = infoList[0].get("mime", "")
            itemTitle = pageData.get("title", "").replace("File:", "")
            if itemUrl and any(m in itemMime for m in ["image/jpeg", "image/png", "image/webp"]):
                matchedImages.append({
                    "title": itemTitle,
                    "url": itemUrl
                })
    except Exception:
        pass

    return matchedImages


def fetchPicsumImage(cacheFolder: Path, imgWidth: int = 1024, imgHeight: int = 1024, seedVal: Optional[str] = None) -> Path:
    if seedVal:
        urlStr = f"https://picsum.photos/seed/{seedVal}/{imgWidth}/{imgHeight}"
        fileBase = f"picsum_{seedVal}.png"
    else:
        timestampVal = int(time.time() * 1000)
        urlStr = f"https://picsum.photos/{imgWidth}/{imgHeight}?random={timestampVal}"
        fileBase = f"picsum_{timestampVal}.png"

    targetFile = cacheFolder / fileBase
    return downloadUrl(urlStr, targetFile)


class OnlineImageStream:
    def __init__(self, streamMode: str = "curated", searchQuery: str = "mathematical art", cacheDir: str = "hpc_downloads"):
        self.streamMode = streamMode
        self.searchQuery = searchQuery
        self.cacheDir = Path(cacheDir)
        self.cacheDir.mkdir(parents=True, exist_ok=True)
        self.itemIndex = 0
        self.queueList: List[Dict[str, str]] = []
        self._loadQueue()

    def _loadQueue(self):
        if self.streamMode == "curated":
            self.queueList.extend(CURATED_ONLINE_ARTWORKS)
        elif self.streamMode == "wikimedia":
            wikiResults = fetchWikimediaImages(self.searchQuery, maxCount=20)
            self.queueList.extend(wikiResults if wikiResults else CURATED_ONLINE_ARTWORKS)
        elif self.streamMode == "mixed":
            self.queueList.extend(CURATED_ONLINE_ARTWORKS)
            self.queueList.extend(fetchWikimediaImages(self.searchQuery, maxCount=15))

    def nextImage(self) -> Dict[str, Any]:
        if self.streamMode == "picsum":
            seedStr = hashlib.md5(f"seed_{self.itemIndex}_{time.time()}".encode()).hexdigest()[:8]
            self.itemIndex += 1
            filePath = fetchPicsumImage(self.cacheDir, seedVal=seedStr)
            return {
                "name": f"picsum_{seedStr}",
                "path": filePath,
                "url": f"https://picsum.photos/seed/{seedStr}/1024/1024",
                "title": f"Lorem Picsum {seedStr}"
            }

        if not self.queueList:
            self._loadQueue()

        currItem = self.queueList[self.itemIndex % len(self.queueList)]
        self.itemIndex += 1

        rawUrl = currItem["url"]
        artTitle = currItem["title"]
        urlHash = hashlib.md5(rawUrl.encode()).hexdigest()[:10]
        safeBase = "".join(c if c.isalnum() else "_" for c in artTitle.lower())[:32]
        destFileName = f"{safeBase}_{urlHash}.png"
        destFilePath = self.cacheDir / destFileName

        if not destFilePath.exists():
            destFilePath = downloadUrl(rawUrl, destFilePath)

        return {
            "name": safeBase,
            "path": destFilePath,
            "url": rawUrl,
            "title": artTitle
        }
