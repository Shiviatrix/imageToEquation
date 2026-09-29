import cv2
import subprocess
from pathlib import Path

img_path = Path("/Users/babayaga/image to equation/hamid art/image.png")
img = cv2.imread(str(img_path))
h, w = img.shape[:2]

crop = img[int(h*0.70):h, 0:w]
cv2.imwrite("scratch/crop_test_clean.png", crop)

res = subprocess.run(["tesseract", "scratch/crop_test_clean.png", "stdout", "-l", "equ", "--psm", "6"], capture_output=True, text=True, env={"TESSDATA_PREFIX": "/Users/babayaga/.gemini/antigravity-ide/tessdata"})
print("EQU:", res.stdout)
