import cv2
import numpy as np
import scipy.fftpack
from typing import Dict, Any

def extract_dct_descriptors(image: np.ndarray, num_terms: int = None, energy_threshold: float = 0.99) -> Dict[str, Any]:
    """
    Extracts DCT frequencies from an RGB image.
    If energy_threshold is provided, it retains only the top frequencies that sum to that percentage
    of total spectral energy, providing massive equation compression natively.
    """
    genes = {}
    
    channels = ['R', 'G', 'B']
    h, w, _ = image.shape
    
    genes['suggest_dct'] = True
    
    max_terms_used = 0
    
    for c_idx, channel in enumerate(channels):
        img_c = image[:, :, c_idx].astype(float) / 255.0
        
        # 2D DCT
        dct = scipy.fftpack.dct(scipy.fftpack.dct(img_c.T, norm='ortho').T, norm='ortho')
        
        flat = np.abs(dct).flatten()
        
        # Sort indices by magnitude (descending)
        indices = np.argsort(flat)[::-1]
        
        if energy_threshold is not None:
            # Parseval's theorem: sum of squares
            sorted_energy = flat[indices] ** 2
            
            # The largest term (indices[0]) is overwhelmingly the DC term. We threshold based on AC energy.
            ac_energy = sorted_energy[1:]
            total_ac = np.sum(ac_energy)
            cum_ac = np.cumsum(ac_energy)
            
            # Find how many terms we need to reach threshold
            keep_ac_count = np.searchsorted(cum_ac, total_ac * energy_threshold) + 1
            keep_count = keep_ac_count + 1 # Include DC term
        else:
            keep_count = num_terms if num_terms is not None else len(flat)
            
        # Ensure we don't exceed num_terms if both are provided
        if num_terms is not None:
            keep_count = min(keep_count, num_terms)
            
        max_terms_used = max(max_terms_used, keep_count)
        
        # Take the top keep_count terms
        final_indices = indices[:keep_count]
        
        genes[f'dct_num_terms_{channel}'] = keep_count
        
        for i, idx in enumerate(final_indices):
            u = idx // w
            v = idx % w
            coeff = dct[u, v]
            
            # Proper scaling for IDCT type II ortho
            alpha_u = np.sqrt(1.0 / h) if u == 0 else np.sqrt(2.0 / h)
            alpha_v = np.sqrt(1.0 / w) if v == 0 else np.sqrt(2.0 / w)
            scaled_coeff = coeff * alpha_u * alpha_v
            
            genes[f'dct_{channel}_u_{i}'] = int(u)
            genes[f'dct_{channel}_v_{i}'] = int(v)
            genes[f'dct_{channel}_C_{i}'] = float(scaled_coeff)
            
    genes['dct_num_terms'] = max_terms_used
    return genes
