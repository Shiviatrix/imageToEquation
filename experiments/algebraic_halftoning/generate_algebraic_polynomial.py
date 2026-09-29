import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import Ridge
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from image_to_equation.evaluator import GridEvaluator

def generate_algebraic_polynomial():
    """
    Computes a purely algebraic polynomial projection of an image.
    Uses Ridge Regression to fit a continuous 2D surface to raster pixel data,
    bypassing traditional frequency-domain (DCT) compression.
    """
    print("Loading image for pure polynomial extraction...")
    
    # Resolve paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "assets", "einstein_matched.jpg")
    
    img = Image.open(img_path).convert('L')
    res = 128
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    
    # Target brightness (0 to 1)
    target = np.array(img).astype(np.float32) / 255.0
    
    # Coordinate grid mapped mathematically from [-1, 1]
    y_lin = np.linspace(-1, 1, res)
    x_lin = np.linspace(-1, 1, res)
    Y_grid, X_grid = np.meshgrid(y_lin, x_lin, indexing='ij')
    
    X_flat = X_grid.flatten()
    Y_flat = Y_grid.flatten()
    Target_flat = target_image.flatten()
    
    # Increase mathematical resolution by pushing polynomial degree
    degree = 40
    print(f"Fitting degree {degree} 2D polynomial...")
    
    terms_list = []
    for d in range(degree + 1):
        for x_pow in range(d + 1):
            y_pow = d - x_pow
            terms_list.append((x_pow, y_pow))
            
    print(f"Number of polynomial terms: {len(terms_list)}")
    
    # Build design matrix A for regression
    A = np.zeros((len(X_flat), len(terms_list)))
    for idx, (p_x, p_y) in enumerate(terms_list):
        A[:, idx] = (X_flat ** p_x) * (Y_flat ** p_y)
        
    print("Running Ridge Regression to stabilize high-degree coefficients...")
    # Ridge regression prevents catastrophic numerical instability at D=40
    # by gently shrinking wildly oscillating coefficients.
    model = Ridge(alpha=1e-7, fit_intercept=False, solver='svd')
    model.fit(A, Target_flat)
    poly_coeffs = model.coef_
    
    print("Constructing AST string...")
    ast_terms = []
    for idx, (p_x, p_y) in enumerate(terms_list):
        coeff = poly_coeffs[idx]
        
        # Prune dead nodes from the AST to save bytes
        if abs(coeff) < 1e-6:
            continue
            
        term = f"{coeff:.6f}"
        if p_x > 0:
            if p_x == 1: term += "*x"
            else: term += f"*(x^{p_x})"
        if p_y > 0:
            if p_y == 1: term += "*y"
            else: term += f"*(y^{p_y})"
            
        ast_terms.append(term)
        
    poly_expr = " + ".join(ast_terms)
    
    # No spirals, just the pure polynomial evaluated as a color!
    # We map the continuous polynomial surface into a grayscale rgb(b, b, b) vector
    full_ast = f"""
    let(x, x*2-1,
      let(y, y*2-1,
        let(brightness, clamp({poly_expr}, 0.0, 1.0),
          rgb(brightness, brightness, brightness)
        )
      )
    )
    """
    
    full_ast = full_ast.replace("\n", "").replace(" ", "").strip()
    print(f"AST Length: {len(full_ast)}")
    
    out_txt = os.path.join(base_dir, "outputs", "pure_poly_ast.txt")
    with open(out_txt, "w") as f:
        f.write(full_ast)
        
    print("Rendering...")
    evaluator = GridEvaluator()
    try:
        img_rgb = evaluator.evaluate(full_ast, 800, 800, render_mode='linear')
        out_png = os.path.join(base_dir, "outputs", "rendered_images", "einstein_pure_poly.png")
        plt.imsave(out_png, img_rgb.numpy())
        print(f"Success! Saved to {out_png}")
    except Exception as e:
        print(f"Error evaluating AST: {e}")

if __name__ == '__main__':
    generate_algebraic_polynomial()
