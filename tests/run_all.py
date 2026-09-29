import cv2
import numpy as np
import os
import glob
import torch
# # from genetic_algo import GeneticAlgorithm
from image_to_equation.evaluator import GridEvaluator
# # from gd_optimizer import full_optimization_pipeline
from image_classifier import ImageTypeClassifier
from fourier_extractor import extract_fourier_descriptors
from image_to_equation.dct_extractor import extract_dct_descriptors
from image_to_equation.symbolic_fitter import FitConfig, binary_iou, fit_binary_mask
from image_to_equation.multiscale_fitter import MultiScaleFitConfig, fit_multiscale_image, psnr, ssim


def _foreground_colour(image: np.ndarray, mask: torch.Tensor) -> tuple[float, float, float]:
    """Return the robust RGB layer colour for an accepted flat-colour mask."""
    foreground = mask.detach().cpu().numpy().astype(bool)
    pixels = image[foreground]
    if len(pixels) == 0:
        return (1.0, 1.0, 1.0)
    return tuple((np.median(pixels, axis=0) / 255.0).clip(0.0, 1.0))


def _background_colour(image: np.ndarray) -> tuple[float, float, float]:
    """Estimate the flat canvas colour from its four corners."""
    height, width = image.shape[:2]
    corners = np.stack((image[0, 0], image[0, width - 1], image[height - 1, 0], image[height - 1, width - 1]))
    return tuple((np.median(corners, axis=0) / 255.0).clip(0.0, 1.0))


def _rgb_expression(colour: tuple[float, float, float]) -> str:
    return "rgb(" + ",".join(f"{channel:.4g}" for channel in colour) + ")"


def _try_symbolic_path(target_img: np.ndarray, evaluator: GridEvaluator):
    """Fit and verify a concise SDF program, otherwise return ``None``.

    The fitter decides from a border-colour segmentation whether the subject is
    explainable by a few typed primitives.  This gate verifies the *emitted*
    evaluator expression rather than trusting the fitter's internal renderer;
    photographs, textured images, and overlong programs fall through to the
    existing adaptive/DCT pipeline unchanged.
    """
    fit = fit_binary_mask(target_img, FitConfig(max_shapes=6))
    foreground_fraction = float(fit.target.float().mean().detach().cpu())
    if not (0.005 <= foreground_fraction <= 0.70):
        return None
    if fit.iou < 0.80 or not fit.program.primitives:
        return None

    height, width = target_img.shape[:2]
    mask_equation = fit.program.to_current_evaluator_expression(height, width)
    if mask_equation is None:
        return None
    # A silhouette layer is not intrinsically a black-background image.  Keep
    # the canvas colour in the procedural payload and composite through the
    # same continuous alpha used by the fitter.
    equation = "over(" + ",".join((
        _rgb_expression(_background_colour(target_img)),
        _rgb_expression(_foreground_colour(target_img, fit.target)),
        mask_equation,
    )) + ")"
    if len(equation) >= 1000:
        return None

    # Verify alpha independently of colour: a legitimate dark subject should
    # not fail the binary gate merely because its red channel is below 0.5.
    mask_rendered = evaluator.evaluate(mask_equation, width, height, render_mode='linear')
    emitted_alpha = mask_rendered[..., 0].float() / 255.0
    emitted_iou = binary_iou(emitted_alpha, fit.target)
    if emitted_iou < 0.80:
        return None
    rendered = evaluator.evaluate(equation, width, height, render_mode='linear')

    return {
        'equation': equation,
        'rendered': rendered.cpu().numpy(),
        'iou': emitted_iou,
        'source_length': len(equation),
        'mdl_bits': fit.mdl_bits,
        'primitive_kinds': [primitive.kind for primitive in fit.program.primitives],
    }


def _try_multiscale_procedural_path(
    target_img: np.ndarray,
    evaluator: GridEvaluator,
    min_psnr: float = 28.0,
    min_ssim: float = 0.90,
):
    """Fit, round-trip, and strictly gate a continuous full-colour scene.

    This is deliberately separate from the silhouette route.  It considers a
    bounded RGB bilinear/radial/SDF program, verifies the *compiled* expression
    against the input, and declines an image that needs more photographic
    entropy than this current procedural vocabulary can explain.  The returned
    source contains only continuous scalar functions—never a residual grid.
    """
    result = fit_multiscale_image(
        target_img,
        MultiScaleFitConfig(
            pyramid_long_sides=(48, 72, 96),
            layers_per_scale=1,
            max_layers=4,
            max_radials=1,
            refine_steps=14,
            max_source_length=6000,
            device=str(evaluator.device),
        ),
    )
    height, width = target_img.shape[:2]
    equation = result.program.to_current_evaluator_expression(height, width)
    if len(equation) > 6000:
        return None

    rendered = evaluator.evaluate(
        equation, width, height, differentiable=True, render_mode='linear'
    ).float() / 255.0
    target = torch.as_tensor(target_img, device=evaluator.device, dtype=torch.float32) / 255.0
    verified_psnr = psnr(rendered, target)
    verified_ssim = ssim(rendered, target)
    if verified_psnr < min_psnr or verified_ssim < min_ssim:
        return None

    preview = (rendered.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return {
        'equation': equation,
        'rendered': preview,
        'psnr': verified_psnr,
        'ssim': verified_ssim,
        'source_length': len(equation),
        'model_bits': result.model_bits,
        'stop_reason': result.stop_reason,
        'accepted_layers': result.accepted_layers,
    }

def process_image(target_path: str):
    print(f"\n" + "="*80)
    print(f"PROCESSING: {target_path}")
    print("="*80)
    
    if not os.path.exists(target_path):
        print(f"Error: {target_path} not found.")
        return
        
    target_img = cv2.imread(target_path)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    h, w = target_img.shape[:2]
    max_dim = 256
    if h > w:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    else:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    target_img = cv2.resize(target_img, (new_w, new_h))
        
    print(f"Loaded target image shape: {target_img.shape}")

    evaluator = GridEvaluator()
    print("\n[Symbolic MDL] Testing concise SDF/CSG reconstruction...")
    symbolic = _try_symbolic_path(target_img, evaluator)
    if symbolic is not None:
        comparison = np.hstack((target_img, symbolic['rendered']))
        out_name = f"result_{os.path.basename(target_path).replace(' ', '_').split('.')[0]}.png"
        cv2.imwrite(out_name, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(
            f"  Accepted {symbolic['primitive_kinds']} | IoU={symbolic['iou']:.4f} | "
            f"chars={symbolic['source_length']} | MDL={symbolic['mdl_bits']:.1f} bits"
        )
        print(f"Saved {out_name}")
        return 1.0 - symbolic['iou'], symbolic['equation'], {
            'mode': 'symbolic_sdf',
            'iou': symbolic['iou'],
            'source_length': symbolic['source_length'],
            'mdl_bits': symbolic['mdl_bits'],
            'primitive_kinds': symbolic['primitive_kinds'],
        }
    print("  Rejected by executable IoU/length gate; retaining adaptive fallback.")

    print("\n[Procedural Multi-Scale] Testing continuous RGB scene reconstruction...")
    multiscale = _try_multiscale_procedural_path(target_img, evaluator)
    if multiscale is not None:
        comparison = np.hstack((target_img, multiscale['rendered']))
        out_name = f"result_{os.path.basename(target_path).replace(' ', '_').split('.')[0]}.png"
        cv2.imwrite(out_name, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(
            f"  Accepted {len(multiscale['accepted_layers'])} continuous layers | "
            f"PSNR={multiscale['psnr']:.2f} dB | SSIM={multiscale['ssim']:.4f} | "
            f"chars={multiscale['source_length']}"
        )
        print(f"Saved {out_name}")
        return 1.0 - multiscale['ssim'], multiscale['equation'], {
            'mode': 'multiscale_procedural',
            'psnr': multiscale['psnr'],
            'ssim': multiscale['ssim'],
            'source_length': multiscale['source_length'],
            'model_bits': multiscale['model_bits'],
            'stop_reason': multiscale['stop_reason'],
        }
    print("  Rejected by strict PSNR/SSIM gate; retaining adaptive fallback.")
    
    print("\n[Phase 4 Enhanced] Classifying image type...")
    classifier = ImageTypeClassifier()
    classification = classifier.classify(target_img)
    image_type = classification['primary_type']
    
    print(f"  Type: {image_type} (confidence: {classification['confidence']:.2f})")
    print(f"  Characteristics: {classification['characteristics']}")
    
    print("  Extracting Fourier descriptors (if contour is found)...")
    fourier_genes = extract_fourier_descriptors(target_img, num_harmonics=10)
    
    max_terms = target_img.shape[0] * target_img.shape[1]
    print(f"  Extracting 2D DCT with 95% energy threshold for massive compression...")
    dct_genes = extract_dct_descriptors(target_img, num_terms=None, energy_threshold=0.95)
    fourier_genes.update(dct_genes)
    
    # Store final width and height for binary encoding
    fourier_genes['width'] = target_img.shape[1]
    fourier_genes['height'] = target_img.shape[0]
    
    if fourier_genes.get('suggest_dct', False):
        print(f"\n[Phase 2.5] Skipping GA and GD for DCT (deterministic).")
        # # from genetic_algo import AdaptiveTemplateFactory
        # template = AdaptiveTemplateFactory._create_dct_template(fourier_genes)
        # eq = template.to_equation_string()
        # Fast analytic reconstruction bypassing PyTorch AST parser for 150,000-term equations
        import scipy.fftpack
        rendered_np = np.zeros_like(target_img)
        h, w = target_img.shape[:2]
        for c_idx, channel in enumerate(['R', 'G', 'B']):
            dct_mat = np.zeros((h, w), dtype=float)
            num_terms = fourier_genes['dct_num_terms']
            for i in range(num_terms):
                u = fourier_genes.get(f'dct_{channel}_u_{i}', 0)
                v = fourier_genes.get(f'dct_{channel}_v_{i}', 0)
                C = fourier_genes.get(f'dct_{channel}_C_{i}', 0.0)
                if abs(C) > 1e-6:
                    alpha_u = np.sqrt(1.0 / h) if u == 0 else np.sqrt(2.0 / h)
                    alpha_v = np.sqrt(1.0 / w) if v == 0 else np.sqrt(2.0 / w)
                    dct_mat[u, v] = C / (alpha_u * alpha_v)
            rendered_c = scipy.fftpack.idct(scipy.fftpack.idct(dct_mat.T, norm='ortho').T, norm='ortho')
            # clamp max 1 min 0 as per equation
            rendered_np[:, :, c_idx] = np.clip(rendered_c * 255.0, 0, 255).astype(np.uint8)
        
        comparison = np.hstack((target_img, rendered_np))
        result_filename = f"result_{os.path.basename(target_path).split('.')[0]}.png"
        cv2.imwrite(result_filename, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"Saved {result_filename}")
        return 0.0, eq, {'mode': 'dct', 'metadata': fourier_genes}
    
    print(f"\n[Phase 2.5] Running Genetic Algorithm (Adaptive Template: {image_type})...")
    ga = GeneticAlgorithm(target_image=target_img, image_type=image_type, fourier_genes=fourier_genes)
    ga_best, history = ga.evolve(generations=15) # 15 gen for better structure
    
    print(f"\nGA Best Fitness: {ga_best.fitness:.2f}")
    
    print("\n[Phase 3] Starting Gradient Descent Optimization...")
    optimized_best = full_optimization_pipeline(
        ga_best, target_img, evaluator, num_restarts=1, max_iterations=150
    )
    
    print(f"\nFinal Optimized Loss: {-optimized_best.fitness:.6f}")
    
    best_tensor = evaluator.evaluate(optimized_best.equation_string, target_img.shape[1], target_img.shape[0])
    best_img = best_tensor.cpu().numpy()
    
    comparison = np.hstack((target_img, best_img))
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    
    out_name = f"result_{os.path.basename(target_path).replace(' ', '_').split('.')[0]}.png"
    cv2.imwrite(out_name, comparison_bgr)
    print(f"Saved {out_name}")
    
    return -optimized_best.fitness, optimized_best.equation_string, {'mode': 'legacy_ga_gd'}

def main():
    images = glob.glob("*.jpg")
    if not images:
        print("No .jpg files found to process.")
        return
        
    images.sort()
    
    results = {}
    for img in images:
        try:
            loss, eq, metadata = process_image(img)
            results[img] = {'loss': loss, 'eq': eq, 'metadata': metadata}
        except Exception as e:
            print(f"Failed processing {img}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE")
    print("="*80)
    for img, data in results.items():
        print(f"{img}: Loss = {data['loss']:.6f}")

if __name__ == "__main__":
    main()
