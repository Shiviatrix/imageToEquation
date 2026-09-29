# Project Results

This document explains how the ImageToEquation project evolved from trying to use AI, to successfully using exact math to turn images into equations.

## Phase 1: Machine Learning (Old Method)

At first, we tried to train an AI model to look at an image and guess the math equation for it.

### Results
- The AI was able to draw simple shapes.
- **The Problem:** It failed when trying to draw complex, detailed photos (like a picture of a face). The AI just couldn't guess long enough or accurate enough equations. 
- **The Workaround:** To fix this, the old system cheated by using standard image compression (like what JPEGs use) for detailed photos. But this defeated the whole purpose of the project, which was to create pure math art.
- *Status: We deleted this old AI code because it didn't work well.*

---

## Phase 2: Exact Math (New Method)

To fix the problems with the AI, we proved that you can use exact math calculations to turn the image into a formula directly, without needing any AI guessing. We created two ways to do this:

### 1. The Polynomial Method
- **How it works:** This calculates a massive algebraic curve (a 40th-degree polynomial) that perfectly matches the brightness of the original image, and then applies a simple wave pattern to draw it.
- **Result:** It creates beautiful, smooth math art without cheating or using standard image compression.

### 2. The Box-Splitting Method (QuadTree)
- **How it works:** This looks at the image and splits it into smaller and smaller boxes. If a part of the image has lots of detail, it splits it into tiny boxes. If it's just a solid color, it leaves it as one big box.
- **Result:** It generates an absolutely massive math formula (over 200,000 characters long) that draws the image using thousands of overlapping boxes. It looks almost exactly like the original photo.

## System Health
- **Code Stability:** The math evaluator can process these massive 200,000-character equations without the program crashing.
- **Integrity:** Every generated equation perfectly follows the strict math rules of the project.
