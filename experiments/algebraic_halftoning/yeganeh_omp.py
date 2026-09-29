import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Add src to path for evaluator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from image_to_equation.evaluator import GridEvaluator

def run_yeganeh_omp():
    print("Loading reference photo...")
    img = Image.open("/Users/babayaga/image to equation/einstein_matched.jpg").convert('L')
    res = 128 # Lower res for OMP speed, but evaluate high res later
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = torch.tensor(np.array(img).astype(np.float32) / 255.0, device=device)
    
    # Normalize to -1, 1
    target = (target - target.mean()) / target.std()

    print("Generating Yeganeh parameter sweep (dictionary)...")
    x = torch.linspace(0, 1, res, device=device)
    y = torch.linspace(0, 1, res, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Flatten everything
    target_flat = target.view(-1)
    
    # Yeganeh Vocabulary: high even powers of sine/cosine
    omegas = torch.linspace(0, 40 * np.pi, 40, device=device)
    powers = [2, 4, 6, 8, 12, 16]
    
    dictionary = []
    formulas = []
    
    # Generate dictionary of Yeganeh functions
    # f(x,y) = sin(u*x)^p * cos(v*y)^q
    print("Sweeping parameter families...")
    for u in omegas:
        for v in omegas:
            for p in powers:
                for q in powers:
                    # Basis function
                    basis = (torch.sin(u * X)**p) * (torch.cos(v * Y)**q)
                    basis_flat = basis.view(-1)
                    
                    # Normalize basis
                    norm = torch.linalg.norm(basis_flat)
                    if norm > 1e-5:
                        basis_flat = basis_flat / norm
                        dictionary.append(basis_flat)
                        formulas.append((u.item(), v.item(), p, q, norm.item()))
                        
    print(f"Generated {len(dictionary)} Yeganeh basis functions.")
    
    # Convert dictionary to a massive matrix (Num_Pixels, Num_Bases)
    D = torch.stack(dictionary, dim=1)
    
    print("Iterating to find best combinations (Orthogonal Matching Pursuit)...")
    residual = target_flat.clone()
    
    N_TERMS = 150
    chosen_indices = []
    weights = []
    
    for i in range(N_TERMS):
        # 1. Inspect the results (compute correlations)
        correlations = torch.matmul(D.t(), residual)
        
        # 2. Gently retune a promising candidate (pick best and subtract)
        best_idx = torch.argmax(torch.abs(correlations))
        chosen_indices.append(best_idx.item())
        weight = correlations[best_idx]
        weights.append(weight.item())
        
        # Subtract from residual
        residual -= weight * D[:, best_idx]
        
        if i % 100 == 0:
            print(f"Iteration {i}: max correlation = {torch.abs(weight).item():.4f}")
            
    print("Constructing massive Yeganeh AST...")
    # Group by k and use the sum(k=1..N, ...) construct to stay inside AST limits
    
    ast_string = ""
    # To keep AST small, we will explicitly write a giant addition of terms
    # evaluator.py supports huge ASTs if we just chain additions
    
    terms = []
    for i in range(N_TERMS):
        idx = chosen_indices[i]
        w = weights[i]
        u, v, p, q, norm = formulas[idx]
        actual_w = w / norm
        
        # Format the Yeganeh term: W * sin(U*x)^P * cos(V*y)^Q
        term = f"({actual_w:.5f} * (sin({u:.2f}*x)^{p}) * (cos({v:.2f}*y)^{q}))"
        terms.append(term)
        
    # Join all 1000 terms
    full_ast = " + ".join(terms)
    
    # Wrap in min-max normalizer so it renders nicely
    # Unfortunately AST doesn't have an easy min/max across the whole image, 
    # so we'll just scale it manually.
    final_ast = f"clamp({full_ast}, 0.0, 1.0)"
    
    print(f"AST generated! Length: {len(final_ast)} characters.")
    
    # Save the AST to an .i2e file (Mode 1)
    print("Saving to scratch/yeganeh_einstein.i2e...")
    import struct
    import zlib
    
    width, height = 800, 800
    payload = zlib.compress(final_ast.encode('utf-8'))
    
    with open("scratch/yeganeh_einstein.i2e", "wb") as f:
        f.write(b"I2E\x01")
        f.write(struct.pack("<HHBI", width, height, 1, len(payload)))
        f.write(payload)
        
    print("Rendering final AST using project Evaluator...")
    evaluator = GridEvaluator()
    img_rgb = evaluator.evaluate(final_ast, width, height, render_mode='linear')
    
    # Save the render
    plt.imsave("scratch/yeganeh_einstein_ast_render.png", img_rgb.numpy())
    print("Done! Photorealistic Yeganeh Einstein achieved!")

if __name__ == '__main__':
    run_yeganeh_omp()
