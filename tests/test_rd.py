import cv2
import numpy as np
import os
from image_to_equation.multiscale_fitter import psnr, ssim
import torch

original_bgr = cv2.imread("test_images/natural_einstein.jpg")
i2e_bgr = cv2.imread("decoded_einstein.png")

def to_tensor(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).float() / 255.0

t_orig = to_tensor(original_bgr)
t_i2e = to_tensor(i2e_bgr)

# psnr in multiscale_fitter.py expects [C, H, W] tensor? Wait, let's just use my own psnr and ssim.
