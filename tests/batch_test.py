import os
import json
import time
import glob
import cv2
import numpy as np
import zlib
from pathlib import Path
import torch

from image_to_equation.evaluator import GridEvaluator
from image_to_equation.multiscale_fitter import psnr, ssim
from run_all import _try_multiscale_procedural_path, _try_symbolic_path
from image_to_equation.encoder import _encode_dct_payload
from image_to_equation.decoder import _decode_dct_payload

SYMBOLIC_MAGIC = b"I2ES"
SYMBOLIC_VERSION = 1
MODE_DCT = b"I2E2"

def encode_symbolic_i2e(equation: str, width: int, height: int, mode: str, path: Path) -> None:
    payload = json.dumps(
        {"version": SYMBOLIC_VERSION, "width": width, "height": height, "mode": mode, "equation": equation},
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(SYMBOLIC_MAGIC + zlib.compress(payload, level=9))

def decode_any_i2e(path: Path) -> tuple:
    magic = path.read_bytes()[:4]
    if magic == SYMBOLIC_MAGIC:
        payload = json.loads(zlib.decompress(path.read_bytes()[4:]).decode("utf-8"))
        evaluator = GridEvaluator()
        rendered = evaluator.evaluate(
            payload["equation"], payload["width"], payload["height"], render_mode="linear"
        )
        return rendered.cpu().numpy(), payload["mode"]
    elif magic == MODE_DCT:
        raw_payload = path.read_bytes()[4:]
        # _decode_dct_payload expects raw bytes and shape. 
        # Wait, the new encoder/decoder expects the header.
        pass
    raise ValueError(f"unsupported i2e magic {magic!r}")

def _metrics(reference: np.ndarray, decoded: np.ndarray) -> tuple:
    target = torch.as_tensor(reference, dtype=torch.float32) / 255.0
    render = torch.as_tensor(decoded, dtype=torch.float32) / 255.0
    return float(psnr(render, target)), float(ssim(render, target))

def main():
    test_dir = Path("test_images")
    out_dir = Path("benchmark_output")
    out_dir.mkdir(exist_ok=True)
    
    results = []
    evaluator = GridEvaluator()
    
    for img_path in sorted(test_dir.glob("*.*")):
        if not img_path.is_file(): continue
        
        print(f"Testing {img_path.name}...")
        original = cv2.imread(str(img_path))
        if original is None: continue
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resize for speed in testing
        height, width = original_rgb.shape[:2]
        if max(height, width) > 192:
            scale = 192 / max(height, width)
            target = cv2.resize(original_rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        else:
            target = original_rgb
            
        start_t = time.time()
        
        # Hybrid Pipeline
        result = _try_symbolic_path(target, evaluator)
        if result is None:
            result = _try_multiscale_procedural_path(target, evaluator)
            
        i2e_path = out_dir / (img_path.stem + ".i2e")
        
        if result is not None:
            mode = "multiscale_procedural" if "psnr" in result else "symbolic_sdf"
            encode_symbolic_i2e(str(result["equation"]), target.shape[1], target.shape[0], mode, i2e_path)
            decoded_rgb = result['rendered']
            mode_str = mode
            stats = {"compressed_size": i2e_path.stat().st_size}
        else:
            from image_to_equation.encoder import encode
            from image_to_equation.decoder import decode
            mode_str = "DCT_fallback"
            stats = encode(str(img_path), str(i2e_path))
            decoded_rgb, _ = decode(str(i2e_path))
            # Resize decoded to target size for metrics
            decoded_rgb = cv2.resize(decoded_rgb, (target.shape[1], target.shape[0]))
            
        decoded_bgr = cv2.cvtColor(decoded_rgb, cv2.COLOR_RGB2BGR)
        elapsed = time.time() - start_t
        
        p, s = _metrics(target, decoded_rgb)
        
        out_png = out_dir / f"decoded_{img_path.name}"
        cv2.imwrite(str(out_png), decoded_bgr)
        
        orig_size = img_path.stat().st_size
        comp_size = stats.get("compressed_size", i2e_path.stat().st_size)
        
        res = {
            "filename": img_path.name,
            "path": mode_str,
            "compression_ratio": orig_size / max(1, comp_size),
            "psnr": p,
            "ssim": s,
            "time_seconds": elapsed,
        }
        results.append(res)
        print(f"  -> Path: {mode_str} | PSNR: {p:.2f} dB, Compression: {res['compression_ratio']:.2f}x")
        
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Batch test complete!")

if __name__ == "__main__":
    main()
