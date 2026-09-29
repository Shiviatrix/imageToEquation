import cv2
import numpy as np
import os
import torch
from genetic_algo import GeneticAlgorithm
from evaluator import GridEvaluator
from gd_optimizer import full_optimization_pipeline
from image_classifier import ImageTypeClassifier
from fourier_extractor import extract_fourier_descriptors
from ga_config import GA_CONFIG
from run_all import _try_multiscale_procedural_path, _try_symbolic_path

def main(target_path="Image.jpg"):
    evaluator = GridEvaluator()
    if not os.path.exists(target_path):
        target_tensor = evaluator.evaluate("cos(10*x) + sin(20*y)", 256, 192)
        target_img = target_tensor.cpu().numpy()
    else:
        target_img = cv2.imread(target_path)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.resize(target_img, (256, 192))
        
    print(f"Loaded target image shape: {target_img.shape}")

    print("\n[Symbolic MDL] Testing concise SDF/CSG reconstruction...")
    symbolic = _try_symbolic_path(target_img, evaluator)
    if symbolic is not None:
        comparison = np.hstack((target_img, symbolic['rendered']))
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite("gd_result_comparison.png", comparison_bgr)
        print(
            f"Accepted {symbolic['primitive_kinds']} | IoU={symbolic['iou']:.4f} | "
            f"chars={symbolic['source_length']} | MDL={symbolic['mdl_bits']:.1f} bits"
        )
        print(f"Final Equation:\n{symbolic['equation']}")
        print("Saved comparison image to gd_result_comparison.png")
        return symbolic
    print("Rejected by executable IoU/length gate; continuing to legacy adaptive path.")

    print("\n[Procedural Multi-Scale] Testing continuous RGB scene reconstruction...")
    multiscale = _try_multiscale_procedural_path(target_img, evaluator)
    if multiscale is not None:
        comparison = np.hstack((target_img, multiscale['rendered']))
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.imwrite("gd_result_comparison.png", comparison_bgr)
        print(
            f"Accepted {len(multiscale['accepted_layers'])} continuous layers | "
            f"PSNR={multiscale['psnr']:.2f} dB | SSIM={multiscale['ssim']:.4f} | "
            f"chars={multiscale['source_length']}"
        )
        print(f"Final Equation:\n{multiscale['equation']}")
        print("Saved comparison image to gd_result_comparison.png")
        return multiscale
    print("Rejected by strict PSNR/SSIM gate; continuing to legacy adaptive path.")
    
    print("\n[Phase 4 Enhanced] Classifying image type...")
    classifier = ImageTypeClassifier()
    classification = classifier.classify(target_img)
    image_type = classification['primary_type']
    
    print(f"  Type: {image_type} (confidence: {classification['confidence']:.2f})")
    print(f"  Characteristics: {classification['characteristics']}")
    
    fourier_genes = None
    if image_type == 'structured_art':
        print("  Extracting Fourier descriptors for exact contour reconstruction...")
        fourier_genes = extract_fourier_descriptors(target_img, num_harmonics=10)
        
    print(f"\n[Phase 2.5] Running Genetic Algorithm (Adaptive Template: {image_type})...")
    ga = GeneticAlgorithm(target_image=target_img, image_type=image_type, fourier_genes=fourier_genes)
    ga_best, history = ga.evolve(generations=10)
    
    print("\n" + "="*50)
    print("GA Evolution Complete")
    print("="*50)
    print(f"Best Fitness: {ga_best.fitness:.2f}")
    
    print("\nStarting Phase 3: Gradient Descent Optimization...")
    optimized_best = full_optimization_pipeline(
        ga_best, target_img, evaluator, num_restarts=1, max_iterations=100
    )
    
    print("\n" + "="*50)
    print("Optimization Complete")
    print("="*50)
    print(f"Final Optimized Loss: {-optimized_best.fitness:.6f}")
    print(f"Final Equation:\n{optimized_best.equation_string}")
    
    best_tensor = evaluator.evaluate(optimized_best.equation_string, target_img.shape[1], target_img.shape[0])
    best_img = best_tensor.cpu().numpy()
    
    comparison = np.hstack((target_img, best_img))
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite("gd_result_comparison.png", comparison_bgr)
    print("Saved comparison image to gd_result_comparison.png")

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "Image.jpg"
    main(img_path)
