import cv2
import subprocess
from pathlib import Path
import os

hamid_dir = Path("/Users/babayaga/image to equation/hamid art")
output_file = Path("scratch/extracted_equations.txt")

with open(output_file, "w") as f:
    for img_path in sorted(hamid_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w = img.shape[:2]
        crop = img[int(h*0.75):h, 0:w]
        
        temp_crop = f"scratch/temp_crop_{img_path.name}"
        cv2.imwrite(temp_crop, crop)
        
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = "/Users/babayaga/.gemini/antigravity-ide/tessdata"
        
        res = subprocess.run(["/opt/homebrew/bin/tesseract", temp_crop, "stdout", "-l", "equ+eng", "--psm", "6"], capture_output=True, text=True, env=env)
        
        f.write(f"=== {img_path.name} ===\n")
        f.write(res.stdout.strip() + "\n\n")
        
        os.remove(temp_crop)

print(f"Finished extracting equations to {output_file}")
