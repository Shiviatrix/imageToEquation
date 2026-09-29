import torch
import numpy as np
import matplotlib.pyplot as plt

def render_flower(filename):
    res = 1200
    x = torch.linspace(-2.5, 2.5, res)
    y = torch.linspace(-2.5, 2.5, res)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Base polar coordinates
    r = torch.sqrt(X**2 + Y**2) + 1e-5
    theta = torch.atan2(Y, X)

    # 1. Coordinate Warping (Yeganeh style)
    # The petals twist slightly as they grow outward
    theta_warped = theta + 0.3 * torch.sin(3 * r)
    
    # 2. Petal Structure (Hierarchical Summation)
    # Layer 1: Large outer petals (e.g., 5 petals)
    petal_R1 = 1.5 + 0.5 * torch.cos(5 * theta_warped) + 0.2 * torch.cos(15 * theta_warped)
    # Layer 2: Inner petals (e.g., 8 petals, rotated)
    petal_R2 = 0.8 + 0.3 * torch.cos(8 * (theta_warped + 0.2)) + 0.1 * torch.cos(24 * theta_warped)

    # 3. Non-linear Exponential Masking
    # e^{-e^{k(r - R)}} creates a sharp, smooth dropoff
    mask_layer1 = torch.exp(-torch.exp(15 * (r - petal_R1)))
    mask_layer2 = torch.exp(-torch.exp(25 * (r - petal_R2)))
    mask_center = torch.exp(-torch.exp(40 * (r - 0.3))) # Stamen mask

    # 4. Organic Textures (High frequency trig interference)
    # Veins flowing along the petals
    veins1 = torch.cos(40 * r + 10 * torch.cos(5 * theta_warped))
    veins2 = torch.cos(60 * r + 15 * torch.cos(8 * theta_warped))
    
    # Stamen texture (Sunflower style Fibonacci grid simulation)
    # cos(A*x + B*y) * cos(A*y - B*x)
    stamen_tex = torch.cos(80*X + 20*Y) * torch.cos(80*Y - 20*X)

    # 5. Composite Procedural Field
    # We combine them additively and multiplicatively
    
    # Base intensity
    intensity = 0.0
    
    # Add Layer 1
    intensity += mask_layer1 * (0.4 + 0.2 * veins1)
    
    # Add Layer 2 (occludes layer 1 slightly where it exists, or just adds)
    intensity += mask_layer2 * (0.6 + 0.2 * veins2)
    
    # Add Center
    intensity += mask_center * (0.8 + 0.4 * stamen_tex)

    # Final polish: clamp and invert for a white background like Yeganeh's sketches
    intensity = torch.clamp(intensity, 0, 1)
    img = 1.0 - intensity.numpy() # Invert so 0 is white, 1 is black

    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

render_flower("scratch/flower_test.png")
print("Rendered Flower test.")
