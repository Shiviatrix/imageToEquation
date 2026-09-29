import torch
import numpy as np
import matplotlib.pyplot as plt

def render_flower_color(filename):
    res = 1200
    x = torch.linspace(-2.5, 2.5, res)
    y = torch.linspace(-2.5, 2.5, res)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Base polar coordinates
    r = torch.sqrt(X**2 + Y**2) + 1e-5
    theta = torch.atan2(Y, X)

    # 1. Coordinate Warping (Yeganeh style)
    theta_warped = theta + 0.3 * torch.sin(3 * r)
    
    # 2. Petal Structure (Hierarchical Summation)
    # Layer 1: Large outer petals (e.g., 5 petals)
    petal_R1 = 1.5 + 0.5 * torch.cos(5 * theta_warped) + 0.2 * torch.cos(15 * theta_warped)
    # Layer 2: Inner petals (e.g., 8 petals, rotated)
    petal_R2 = 0.8 + 0.3 * torch.cos(8 * (theta_warped + 0.2)) + 0.1 * torch.cos(24 * theta_warped)

    # 3. Non-linear Exponential Masking
    mask_layer1 = torch.exp(-torch.exp(15 * (r - petal_R1)))
    mask_layer2 = torch.exp(-torch.exp(25 * (r - petal_R2)))
    mask_center = torch.exp(-torch.exp(40 * (r - 0.3))) # Stamen mask

    # 4. Organic Textures (High frequency trig interference)
    # Veins flowing along the petals
    veins1 = torch.cos(40 * r + 10 * torch.cos(5 * theta_warped))
    veins2 = torch.cos(60 * r + 15 * torch.cos(8 * theta_warped))
    
    # Stamen texture (Sunflower style Fibonacci grid simulation)
    stamen_tex = torch.cos(80*X + 20*Y) * torch.cos(80*Y - 20*X)

    # 5. RGB Composite Procedural Field
    # Shape: (3, res, res)
    img = torch.zeros((3, res, res))
    
    # Background Color (Dark emerald green/blue)
    bg_color = torch.tensor([0.05, 0.15, 0.10]).view(3, 1, 1)
    img += bg_color * (1.0 - mask_layer1)
    
    # Outer Petals (Deep Pink / Magenta)
    outer_color = torch.tensor([0.9, 0.2, 0.5]).view(3, 1, 1)
    # Modulate brightness by veins
    outer_shade = (0.7 + 0.3 * veins1)
    img = img * (1.0 - mask_layer1) + (outer_color * outer_shade) * mask_layer1
    
    # Inner Petals (Vibrant Golden Yellow)
    inner_color = torch.tensor([1.0, 0.8, 0.1]).view(3, 1, 1)
    inner_shade = (0.7 + 0.3 * veins2)
    img = img * (1.0 - mask_layer2) + (inner_color * inner_shade) * mask_layer2
    
    # Center Stamen (Dark Brown/Crimson)
    center_color = torch.tensor([0.3, 0.1, 0.0]).view(3, 1, 1)
    center_shade = (0.5 + 0.5 * stamen_tex)
    img = img * (1.0 - mask_center) + (center_color * center_shade) * mask_center

    # Final polish: Add a global lighting vignette based on radius
    vignette = torch.exp(-0.2 * r)
    img = img * vignette.unsqueeze(0)

    # Clamp to [0, 1]
    img = torch.clamp(img, 0, 1)
    
    # Convert to HxWxC for matplotlib
    img_np = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

render_flower_color("scratch/flower_color_test.png")
print("Rendered Colored Flower test.")
