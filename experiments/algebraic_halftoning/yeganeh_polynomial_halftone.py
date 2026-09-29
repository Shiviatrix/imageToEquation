import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from image_to_equation.evaluator import GridEvaluator

def run_poly_halftone():
    print("Loading image for polynomial extraction...")
    img = Image.open("/Users/babayaga/image to equation/einstein_matched.jpg").convert('L')
    res = 100
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    
    # Target brightness (0 to 1)
    target = np.array(img).astype(np.float32) / 255.0
    
    # Coordinate grid [-1, 1]
    y_lin = np.linspace(-1, 1, res)
    x_lin = np.linspace(-1, 1, res)
    Y, X = np.meshgrid(y_lin, x_lin, indexing='ij')
    
    # Flatten
    X_f = X.flatten()
    Y_f = Y.flatten()
    T_f = target.flatten()
    
    # We want a 2D polynomial of degree D
    D = 25
    print(f"Fitting degree {D} 2D polynomial...")
    
    terms_list = []
    for d in range(D + 1):
        for i in range(d + 1):
            j = d - i
            terms_list.append((i, j))
            
    print(f"Number of polynomial terms: {len(terms_list)}")
    
    # Build design matrix
    A = np.zeros((len(X_f), len(terms_list)))
    for idx, (p_x, p_y) in enumerate(terms_list):
        A[:, idx] = (X_f ** p_x) * (Y_f ** p_y)
        
    # Least squares fit (NOT gradient descent!)
    coeffs, _, _, _ = np.linalg.lstsq(A, T_f, rcond=None)
    
    print("Constructing AST string...")
    ast_terms = []
    for idx, (p_x, p_y) in enumerate(terms_list):
        c = coeffs[idx]
        if abs(c) < 1e-4:
            continue
            
        term = f"{c:.5f}"
        if p_x > 0:
            if p_x == 1: term += "*x"
            else: term += f"*(x^{p_x})"
        if p_y > 0:
            if p_y == 1: term += "*y"
            else: term += f"*(y^{p_y})"
            
        ast_terms.append(term)
        
    poly_expr = " + ".join(ast_terms)
    
    # Yeganeh style fingerprint masking
    # The image x,y in evaluator go from 0 to 1.
    # Our polynomial used -1 to 1. We need to map evaluator x,y (0 to 1) to -1 to 1.
    # EvalX = (x * 2 - 1), EvalY = (y * 2 - 1)
    # But wait, AST uses x and y. So let's wrap it in a `let`.
    
    # Let's create the final AST!
    # A fingerprint pattern is cos( freq * sqrt(X^2 + Y^2) )
    # Halftone thresholding: smoothstep
    
    full_ast = f"""
    let(x, x*2-1,
      let(y, y*2-1,
        let(brightness, clamp({poly_expr}, 0.0, 1.0),
          let(pattern, cos(200 * sqrt(x^2 + y^2)),
            sdfmask(pattern - (brightness*2-1), 0.1)
          )
        )
      )
    )
    """
    
    full_ast = full_ast.replace("\\n", "").replace(" ", "").strip()
    print(f"AST Length: {len(full_ast)}")
    
    print("Rendering...")
    evaluator = GridEvaluator()
    try:
        img_rgb = evaluator.evaluate(full_ast, 800, 800, render_mode='linear')
        plt.imsave("scratch/einstein_fingerprint.png", img_rgb.numpy())
        print("Success!")
    except Exception as e:
        print(f"Error evaluating AST: {e}")

if __name__ == '__main__':
    run_poly_halftone()
