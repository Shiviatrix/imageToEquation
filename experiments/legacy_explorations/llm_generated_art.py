import torch
import numpy as np
import matplotlib.pyplot as plt

def render_art():
    # Initialize spatial grid
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 1000), torch.linspace(-1, 1, 1000))
    R, Theta = torch.atan2(Y, X), torch.sqrt(X**2 + Y**2)

    # Compute base coordinates
    X = (X + 1) / 2
    Y = (Y + 1) / 2
    R = R * 10
    Theta = Theta * 2 * np.pi

    # Compute geometric height profile
    k = 1000
    threshold = 0.5
    M = torch.exp(-torch.exp(k * (R - threshold)))
    Height = M * torch.cos(Theta)

    # Update Z-buffer
    H_map = torch.zeros_like(Height)
    LayerID_map = torch.zeros_like(Height)
    H_map[Height > H_map] = Height[Height > H_map]
    LayerID_map[Height > H_map] = torch.max(LayerID_map[Height > H_map], torch.ones_like(Height[Height > H_map]))

    # Map RGB colors and textural trig functions
    H_0 = torch.exp(-torch.exp(100 * (R - 0.5)))
    H_1 = torch.exp(-torch.exp(100 * (R - 0.75)))
    H_2 = torch.exp(-torch.exp(100 * (R - 1.0)))
    Color = torch.stack([H_0, H_1, H_2], dim=-1)

    # Compute lighting
    N = torch.stack([-torch.sin(Theta), -torch.cos(Theta), 1], dim=-1)
    L = torch.tensor([0.5, 0.5, 0.5])
    Diffuse = torch.max(N.dot(L), 0)
    Specular = torch.pow(torch.max(N.dot(L), 0), 50)
    Ambient = 0.1

    # Final composite
    Final = Color * (Diffuse + Ambient) + Specular
    Final = 1 - torch.exp(-Final * 100)
    Final = Final.clamp(0, 1)

    # Output to a 3-channel RGB image tensor
    plt.imshow(Final.numpy(), cmap='viridis')
    plt.axis('off')
    plt.savefig('scratch/llm_art_output.png')

render_art()