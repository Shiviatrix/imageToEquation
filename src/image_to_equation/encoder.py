import io
import struct
import zlib
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch

from image_to_equation.neural_symbolic import ImageToAST, YeganehVocabulary, program_to_source
from image_to_equation.evaluator import GridEvaluator
from image_to_equation.multiscale_fitter import psnr, ssim
from image_to_equation.dct_extractor import extract_dct_descriptors


# Magic bytes: 'I2E' + version 1
I2E_MAGIC = b'I2E\x01'
MODE_SYMBOLIC = 0x01
MODE_DCT = 0x02

@torch.no_grad()
def _try_symbolic_encode(
    image_bgr: np.ndarray,
    model_path: Path,
    min_psnr: float = -1000.0,
    max_length: int = 1500,
) -> Optional[str]:
    """Attempts to encode the image using the Neural-Symbolic AST model."""
    if not model_path.exists():
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vocabulary = YeganehVocabulary(constant_bins=int(checkpoint["vocabulary_bins"]))
    model = ImageToAST(vocabulary=vocabulary).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 64))
    
    # Preprocess
    original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    target = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_AREA)
    model_input = torch.from_numpy(target).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    # Generate candidates
    candidates = model.generate(model_input, beam_size=4)
    evaluator = GridEvaluator()
    target_tensor = torch.from_numpy(target).float().div(255.0)
    
    best_source = None
    best_psnr = -1.0
    
    for program in candidates:
        source = program_to_source(program, model.vocabulary)
        if len(source) > max_length:
            continue
            
        decoded = evaluator.evaluate(source, target.shape[1], target.shape[0], render_mode="linear").cpu().numpy()
        decoded_tensor = torch.from_numpy(decoded).float().div(255.0)
        
        current_psnr = float(psnr(decoded_tensor, target_tensor))
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_source = source
            
    if best_psnr >= min_psnr and best_source is not None:
        return best_source
        
    return None

def _encode_dct_payload(image_rgb: np.ndarray, energy_threshold: float = 0.999) -> bytes:
    """Encodes an image to the raw DCT fallback payload structure."""
    genes = extract_dct_descriptors(image_rgb, num_terms=None, energy_threshold=energy_threshold)
    
    inner = io.BytesIO()
    channels = ['R', 'G', 'B']
    
    for channel in channels:
        num_terms = genes.get(f'dct_num_terms_{channel}', 0)
        inner.write(struct.pack('<I', num_terms))
        
        terms = []
        for i in range(num_terms):
            u = genes.get(f'dct_{channel}_u_{i}', 0)
            v = genes.get(f'dct_{channel}_v_{i}', 0)
            C = genes.get(f'dct_{channel}_C_{i}', 0.0)
            terms.append((u, v, C))
            
        # Sort terms for optimal delta compression
        terms.sort(key=lambda x: (x[0], x[1]))
        prev_u = prev_v = 0
        
        for u, v, C in terms:
            delta_u = u - prev_u
            delta_v = v - prev_v
            inner.write(struct.pack('<hhf', delta_u, delta_v, C))
            prev_u, prev_v = u, v
            
    return inner.getvalue()

def encode(image_path: str, output_path: str, model_path: Optional[Path] = None) -> dict:
    """
    Main reference encoder for the .i2e format.
    Automatically routes between Symbolic AST and DCT Fallback.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    height, width = image_bgr.shape[:2]
    
    # 1. Attempt Neural-Symbolic Encoding
    ast_source = None
    if model_path:
        ast_source = _try_symbolic_encode(image_bgr, model_path)
        
    out = io.BytesIO()
    out.write(I2E_MAGIC)
    out.write(struct.pack('<HH', width, height))
    
    if ast_source is not None:
        # Symbolic Mode
        out.write(struct.pack('<B', MODE_SYMBOLIC))
        raw_payload = ast_source.encode('utf-8')
        mode_str = "Symbolic AST"
    else:
        # DCT Fallback Mode
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out.write(struct.pack('<B', MODE_DCT))
        raw_payload = _encode_dct_payload(image_rgb)
        mode_str = "DCT Fallback"
        
    compressed_payload = zlib.compress(raw_payload)
    out.write(struct.pack('<I', len(compressed_payload)))
    out.write(compressed_payload)
    
    with open(output_path, 'wb') as f:
        f.write(out.getvalue())
        
    return {
        "mode": mode_str,
        "width": width,
        "height": height,
        "original_size": image_bgr.nbytes,
        "compressed_size": len(out.getvalue())
    }
