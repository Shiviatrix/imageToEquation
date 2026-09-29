import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def run_dct_extraction():
    # 1. Load image and convert to grayscale
    img_path = "/Users/babayaga/image to equation/einstein_matched.jpg"
    img = Image.open(img_path).convert('L')
    
    # Resize to a manageable resolution for the math equation (e.g. 128x128 or 256x256)
    res = 256
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    img_arr = np.array(img).astype(float) / 255.0

    # 2. Compute 2D DCT
    dct_coeffs = dct2(img_arr)
    
    # 3. Extract the top N largest coefficients
    N_terms = 5000  # 5,000 equations is enough for photorealism at 256x256
    flat_coeffs = np.abs(dct_coeffs.flatten())
    indices = np.argsort(flat_coeffs)[-N_terms:]
    
    u_idx, v_idx = np.unravel_index(indices, dct_coeffs.shape)
    
    # 4. Generate the pure mathematical Python script!
    script_content = f"""import torch
import numpy as np
import matplotlib.pyplot as plt

def render_photoreal_math():
    res = {res}
    print("Evaluating {N_terms} trigonometric equations...")
    
    # Create coordinate grid [0, pi]
    x = torch.linspace(0, np.pi, res)
    y = torch.linspace(0, np.pi, res)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    H = torch.zeros_like(X)
    
    # DCT Basis components
"""
    
    # Append the explicit mathematical terms
    # We batch them for evaluation speed in PyTorch, but conceptually they are pure sum of cosines.
    script_content += "    # Expanding the 2D Fourier Cosine Series\n"
    
    # To avoid generating a 5000 line file which is slow to parse, we will inject the arrays directly,
    # representing: H = sum(C_uv * cos(u*X) * cos(v*Y))
    c_vals = []
    u_vals = []
    v_vals = []
    for u, v in zip(u_idx, v_idx):
        c_vals.append(dct_coeffs[u, v])
        u_vals.append(u)
        v_vals.append(v)
        
    script_content += f"    C = torch.tensor({c_vals}, dtype=torch.float32)\n"
    script_content += f"    U = torch.tensor({u_vals}, dtype=torch.float32)\n"
    script_content += f"    V = torch.tensor({v_vals}, dtype=torch.float32)\n"
    
    script_content += """
    # Evaluate the exact mathematical series
    for i in range(len(C)):
        H += C[i] * torch.cos(U[i] * X) * torch.cos(V[i] * Y)
        
    # Scale back to 0-1
    H = torch.clamp(H, 0, 1)
    
    # Render Output
    img_np = H.numpy()
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('scratch/einstein_dct_render.png', bbox_inches='tight', pad_inches=0, dpi=200, facecolor='black')
    plt.close()

if __name__ == '__main__':
    render_photoreal_math()
"""
    
    with open("scratch/einstein_dct.py", "w") as f:
        f.write(script_content)
        
    print("Successfully encoded Einstein into a pure trigonometric series script!")

if __name__ == "__main__":
    run_dct_extraction()
