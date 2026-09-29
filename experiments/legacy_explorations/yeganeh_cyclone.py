import torch
import numpy as np
import matplotlib.pyplot as plt

def render_cyclone(filename, warp_factor=1.0, texture_freq=20.0):
    # Create grid
    res = 1000
    x = torch.linspace(-1.5, 1.5, res)
    y = torch.linspace(-1.5, 1.5, res)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Polar distance
    P = torch.sqrt(X**2 + Y**2)
    # Prevent log(0)
    P_safe = torch.clamp(P, min=1e-5)

    # Rotation field R(x,y)
    # R(x,y) = (17/20 - 3/10 * exp(-exp(2*(x^2+y^2-1)))) * ln(x^2+y^2)
    term1 = 17.0/20.0
    term2 = (3.0/10.0) * torch.exp(-torch.exp(2.0 * (P**2 - 1.0)))
    R = (term1 - term2) * torch.log(P_safe**2) * warp_factor

    # Warped angular coordinate Q(x,y)
    # Q(x,y) = arctan(1 / (1000 * (1000 * |cos(R)x - sin(R)y| + 1))) * (cos(R)y + sin(R)x)
    cos_R = torch.cos(R)
    sin_R = torch.sin(R)
    
    denom_inner = 1000.0 * torch.abs(cos_R * X - sin_R * Y) + 1.0
    arctan_term = torch.atan(1.0 / (1000.0 * denom_inner))
    
    Q = arctan_term * (cos_R * Y + sin_R * X)

    # Texture field (simplified from the original sum)
    # W(x,y) = cos(freq * Q) * cos(freq * P)
    W = torch.cos(texture_freq * Q * 100.0) * torch.cos(texture_freq * P * 10.0)
    
    # Render
    img = W.numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray', origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

# 1. Baseline Cyclone Warp
render_cyclone("scratch/cyclone_baseline.png", warp_factor=1.0, texture_freq=2.0)

# 2. Stronger Rotation Field (Warp Factor = 3.0)
render_cyclone("scratch/cyclone_strong_warp.png", warp_factor=3.0, texture_freq=2.0)

# 3. Inverted Rotation Field (Warp Factor = -1.0)
render_cyclone("scratch/cyclone_inverted_warp.png", warp_factor=-1.0, texture_freq=2.0)

# 4. Dense Texture
render_cyclone("scratch/cyclone_dense.png", warp_factor=1.0, texture_freq=8.0)

print("Rendered Cyclone tests.")
