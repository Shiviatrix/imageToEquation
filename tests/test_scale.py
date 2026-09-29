import torch
import cv2
import numpy as np
from image_to_equation.evaluator import GridEvaluator

evaluator = GridEvaluator()
# Equation for a sharp circle using evaluator's supported math syntax (^ for power)
eq = "over(rgb(0.2,0.2,0.8),rgb(1.0,0.8,0.2),sdfmask(((x-0.5)^2 + (y-0.5)^2) - 0.05, 0.001))"

# Small 256x256
img_small = evaluator.evaluate(eq, 256, 256, render_mode="linear").cpu().numpy()
img_small_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
cv2.imwrite("circle_256.png", img_small_bgr)

# Large 2560x2560
img_large = evaluator.evaluate(eq, 2560, 2560, render_mode="linear").cpu().numpy()
img_large_bgr = cv2.cvtColor(img_large, cv2.COLOR_RGB2BGR)
cv2.imwrite("circle_2560.png", img_large_bgr)

# Let's crop a 256x256 section of both to show the difference
# For the small one, we resize it to 2560x2560 (nearest neighbor to show pixels) and then crop
img_small_upscaled = cv2.resize(img_small_bgr, (2560, 2560), interpolation=cv2.INTER_NEAREST)
crop_small = img_small_upscaled[1280:1536, 1280:1536]
cv2.imwrite("crop_small.png", crop_small)

crop_large = img_large_bgr[1280:1536, 1280:1536]
cv2.imwrite("crop_large.png", crop_large)
