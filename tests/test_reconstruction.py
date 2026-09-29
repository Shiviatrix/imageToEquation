import cv2
import numpy as np
import torch
import sys
import os

# Add parent dir to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_classifier import ImageTypeClassifier
from genetic_algo import GeneticAlgorithm
from gd_optimizer import full_optimization_pipeline
from evaluator import GridEvaluator
from fourier_extractor import extract_fourier_descriptors

def test_end_to_end_reconstruction():
    target_img = np.zeros((64, 64, 3), dtype=np.uint8)
    
    # A 5-pointed star polygon
    pts = np.array([
        [32, 5], [40, 25], [60, 25], [44, 38], 
        [50, 58], [32, 45], [14, 58], [20, 38], 
        [4, 25], [24, 25]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Draw an orange star (R=255, G=128, B=0)
    cv2.fillPoly(target_img, [pts], (0, 128, 255))
    
    # 2. Extract Fourier Descriptors
    # Force structured_art for the test to ensure we test FourierContour
    image_type = 'structured_art'
    fourier_genes = extract_fourier_descriptors(target_img, num_harmonics=10)
    
    # 3. GA Evolution
    ga = GeneticAlgorithm(
        target_image=target_img, 
        image_type=image_type,
        fourier_genes=fourier_genes
    )
    ga.pop_size = 10
    
    # Force FourierFill for the first seed to ensure we test it
    perfect_seed = ga.population[0]
    perfect_seed.genes['base_primitive'] = 'FourierFill'
    perfect_seed.genes['num_instances'] = 1.0  # Force single instance
    perfect_seed.update_equation()
    
    # DO NOT EVOLVE! The GA might pick FourierContour over FourierFill.
    # We want to test FourierFill directly.
    ga_best = perfect_seed
    
    # 4. Skip GD Optimization to see pure mathematical output
    evaluator = GridEvaluator()
    optimized_best = perfect_seed # Just use the seed directly
    
    # 5. Visualize
    print(f"\nFinal Equation: {optimized_best.equation_string}")
    
    # Render final image for comparison
    rendered_tensor = evaluator.evaluate(optimized_best.equation_string, 64, 64)
    rendered_img = rendered_tensor.cpu().numpy()
    
    # Save comparison image
    comparison = np.hstack((target_img, rendered_img))
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite("synthetic_reconstruction.png", comparison_bgr)
    
if __name__ == "__main__":
    test_end_to_end_reconstruction()
