import struct
import zlib
import io
from typing import Optional

import numpy as np
import scipy.fftpack
from image_to_equation.evaluator import GridEvaluator

I2E_MAGIC = b'I2E\x01'
MODE_SYMBOLIC = 0x01
MODE_DCT = 0x02

def _decode_dct_payload(payload_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Decodes a raw DCT payload bytes object back into an RGB image."""
    f = io.BytesIO(payload_bytes)
    channels = ['R', 'G', 'B']
    genes = {}
    
    for channel in channels:
        num_terms_bytes = f.read(4)
        if not num_terms_bytes:
            break
        num_terms = struct.unpack('<I', num_terms_bytes)[0]
        genes[f'dct_num_terms_{channel}'] = num_terms
        
        prev_u = 0
        prev_v = 0
        
        for i in range(num_terms):
            delta_u, delta_v, C = struct.unpack('<hhf', f.read(8))
            u = prev_u + delta_u
            v = prev_v + delta_v
            
            genes[f'dct_{channel}_u_{i}'] = u
            genes[f'dct_{channel}_v_{i}'] = v
            genes[f'dct_{channel}_C_{i}'] = C
            
            prev_u = u
            prev_v = v
            
    rendered_np = np.zeros((height, width, 3), dtype=np.uint8)
    for c_idx, channel in enumerate(channels):
        dct_mat = np.zeros((height, width), dtype=float)
        num_terms = genes.get(f'dct_num_terms_{channel}', 0)
        
        for i in range(num_terms):
            u = genes[f'dct_{channel}_u_{i}']
            v = genes[f'dct_{channel}_v_{i}']
            C = genes[f'dct_{channel}_C_{i}']
            
            if abs(C) > 1e-6:
                alpha_u = np.sqrt(1.0 / height) if u == 0 else np.sqrt(2.0 / height)
                alpha_v = np.sqrt(1.0 / width) if v == 0 else np.sqrt(2.0 / width)
                dct_mat[u, v] = C / (alpha_u * alpha_v)
                
        rendered_c = scipy.fftpack.idct(scipy.fftpack.idct(dct_mat.T, norm='ortho').T, norm='ortho')
        rendered_np[:, :, c_idx] = np.clip(rendered_c * 255.0, 0, 255).astype(np.uint8)
        
    return rendered_np

def decode(input_path: str) -> tuple[np.ndarray, dict]:
    """
    Main reference decoder for the .i2e format.
    Automatically handles Symbolic AST or DCT Fallback blocks.
    """
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != I2E_MAGIC:
            raise ValueError(f"Invalid or unsupported magic signature in {input_path}")
            
        width, height = struct.unpack('<HH', f.read(4))
        mode = struct.unpack('<B', f.read(1))[0]
        payload_length = struct.unpack('<I', f.read(4))[0]
        
        compressed_payload = f.read(payload_length)
        raw_payload = zlib.decompress(compressed_payload)
        
        mode_str = ""
        source = ""
        
        if mode == MODE_SYMBOLIC:
            mode_str = "Symbolic AST"
            source = raw_payload.decode('utf-8')
            evaluator = GridEvaluator()
            rendered_tensor = evaluator.evaluate(source, width, height, render_mode="linear")
            rendered_np = rendered_tensor.cpu().numpy()
        elif mode == MODE_DCT:
            mode_str = "DCT Fallback"
            rendered_np = _decode_dct_payload(raw_payload, width, height)
        else:
            raise ValueError(f"Unknown payload mode: {mode}")
            
    return rendered_np, {
        "mode": mode_str,
        "width": width,
        "height": height,
        "source": source
    }
