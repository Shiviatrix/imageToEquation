import cv2
import numpy as np
from typing import Dict

def extract_fourier_descriptors(image: np.ndarray, num_harmonics: int = 10) -> Dict[str, float]:
    """
    Extracts Fourier series coefficients for the largest contour in the image.
    Returns a dictionary of genes that can be injected into FourierContour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Invert if the background is white (heuristic: if corner is white, invert)
    if thresh[0, 0] == 255:
        thresh = cv2.bitwise_not(thresh)
        
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        # Fallback if no contours found
        genes = {'fourier_A0': 0.5, 'fourier_C0': 0.5}
        for k in range(1, num_harmonics + 1):
            genes[f'fourier_A_{k}'] = 0.0
            genes[f'fourier_B_{k}'] = 0.0
            genes[f'fourier_C_{k}'] = 0.0
            genes[f'fourier_D_{k}'] = 0.0
        genes['suggest_instancer'] = False
        return genes
        
    # Count significant contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2)
    
    # Extract mean color
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask)
    # The image is presumably RGB, so mean_val is (R, G, B, _)
    # Multiply by 0.99 to avoid hitting the sharp exp cut-off at x=1.0 in yeganeh_clamp
    color_R = (mean_val[0] / 255.0) * 0.99
    color_G = (mean_val[1] / 255.0) * 0.99
    color_B = (mean_val[2] / 255.0) * 0.99
    
    # Normalize coordinates to [0, 1] for our GridEvaluator
    h, w = image.shape[:2]
    x = points[:, 0] / w
    y = points[:, 1] / h
    
    # Complex array Z = X + iY
    Z = x + 1j * y
    
    # FFT
    fft = np.fft.fft(Z)
    N = len(Z)
    
    genes = {}
    
    # DC component (k=0)
    genes['fourier_A0'] = fft[0].real / N
    genes['fourier_C0'] = fft[0].imag / N
    
    # Check if this image has many distinct objects
    genes['suggest_instancer'] = len(significant_contours) > 3
    
    # Harmonics (k > 0)
    # We take the first `num_harmonics` frequencies.
    # For a real signal X, X_k and X_{-k} are conjugates. But our signal Z is complex,
    # so we need both positive and negative frequencies.
    # To simplify, we can reconstruct X(t) and Y(t) as real Fourier series directly from x and y!
    
    fft_x = np.fft.fft(x) / N
    fft_y = np.fft.fft(y) / N
    
    # For a real signal, FFT[k] = (A_k - i B_k)/2.
    # So A_k = 2 * Re(FFT[k]), B_k = -2 * Im(FFT[k])
    
    genes['fourier_A0'] = fft_x[0].real
    genes['fourier_C0'] = fft_y[0].real
    
    for k in range(1, num_harmonics + 1):
        if k < N // 2:
            genes[f'fourier_A_{k}'] = 2 * fft_x[k].real
            genes[f'fourier_B_{k}'] = -2 * fft_x[k].imag
            
            genes[f'fourier_C_{k}'] = 2 * fft_y[k].real
            genes[f'fourier_D_{k}'] = -2 * fft_y[k].imag
        else:
            genes[f'fourier_A_{k}'] = 0.0
            genes[f'fourier_B_{k}'] = 0.0
            genes[f'fourier_C_{k}'] = 0.0
            genes[f'fourier_D_{k}'] = 0.0
            
    genes['color_R'] = color_R
    genes['color_G'] = color_G
    genes['color_B'] = color_B
    
    return genes
