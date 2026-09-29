import torch
import numpy as np
import matplotlib.pyplot as plt

def fbm(x, y, octaves=6):
    val = torch.zeros_like(x)
    amp = 1.0
    freq = 1.0
    for i in range(octaves):
        theta = float(i) * 2.39996
        cx = np.cos(theta)
        cy = np.sin(theta)
        nx = x * cx - y * cy
        ny = x * cy + y * cx
        val += amp * torch.sin(freq * nx + float(i)*1.13) * torch.cos(freq * ny - float(i)*0.77)
        amp *= 0.5
        freq *= 2.1
    return val

def render_photoreal(filename):
    res = 1200
    x = torch.linspace(-1.5, 1.5, res)
    y = torch.linspace(-1.5, 1.5, res)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Organic warp to break perfect symmetry
    noise_warp1 = fbm(X*2, Y*2, octaves=4)
    noise_warp2 = fbm(X*2+10, Y*2-10, octaves=4)
    X_w = X + 0.08 * noise_warp1
    Y_w = Y + 0.08 * noise_warp2

    r = torch.sqrt(X_w**2 + Y_w**2) + 1e-5
    theta = torch.atan2(Y_w, X_w)

    # Height map and color map
    H = torch.full_like(r, -10.0) # Start with very low ground
    color_map = torch.zeros((3, res, res))

    # Petals (spiraling outwards like a rose)
    num_petals = 35
    golden_angle = 137.5 * np.pi / 180.0
    
    for i in range(num_petals):
        # Petal parameters
        p_theta = i * golden_angle
        # Petals get larger and further out as they go
        p_radius = 0.05 + (i / num_petals)**1.2 * 0.9 
        p_width = 0.3 + 0.4 * (i / num_petals)
        p_length = 0.3 + 0.4 * (i / num_petals)
        
        # Angular distance to petal center
        d_theta = (theta - p_theta) % (2 * np.pi)
        d_theta = torch.where(d_theta > np.pi, 2 * np.pi - d_theta, d_theta)
        
        # Petal shape profiles (Parabolic arches)
        width_profile = torch.clamp(1.0 - (d_theta / p_width)**2, 0, 1)
        length_profile = torch.clamp(1.0 - ((r - p_radius) / p_length)**2, 0, 1)
        
        # Petal base height (increases slightly for overlapping)
        # Petals further out (higher i) droop downwards
        droop = -0.3 * (r - 0.2)**2
        petal_h = 0.1 * (i / num_petals) + 0.3 * width_profile * length_profile + droop
        
        # Add high-frequency noise for organic wrinkly petal texture
        wrinkles = 0.02 * fbm(X*20 + i, Y*20 - i, octaves=3)
        petal_h += wrinkles * width_profile * length_profile
        
        # Mask for this petal
        petal_mask = (width_profile > 0) & (length_profile > 0)
        
        # Z-buffer update: only draw if this petal is higher than what's already there
        update_mask = (petal_h > H) & petal_mask
        H = torch.where(update_mask, petal_h, H)
        
        # Color gradient: Deep crimson in center to velvety red/pink on edges
        center_color = torch.tensor([0.4, 0.0, 0.05]).view(3,1,1)
        edge_color = torch.tensor([0.9, 0.1, 0.2]).view(3,1,1)
        blend = torch.clamp((r / 1.5), 0, 1)
        p_color = center_color * (1 - blend) + edge_color * blend
        
        # Add subtle vein discoloration
        veins = torch.clamp(torch.cos(60 * r + 15 * torch.cos(10 * theta)), 0, 1)
        p_color -= 0.05 * veins
        
        # Apply color to visible pixels
        for c in range(3):
            color_map[c] = torch.where(update_mask, p_color[c] * torch.ones_like(r), color_map[c])

    # Compute Normals from Height Map (Finite Differences)
    dHdx = torch.zeros_like(H)
    dHdy = torch.zeros_like(H)
    dHdx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / 2.0
    dHdy[1:-1, :] = (H[2:, :] - H[:-2, :]) / 2.0
    
    # Scale controls physical bumpiness of the normal map
    bump_scale = 30.0
    Nx = -dHdx * bump_scale
    Ny = -dHdy * bump_scale
    Nz = torch.ones_like(H)
    
    # Normalize
    N_len = torch.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nx /= N_len
    Ny /= N_len
    Nz /= N_len

    # Lighting Setup (Directional Light + Ambient)
    # Light coming from top-left
    Lx, Ly, Lz = -0.5, 0.5, 0.8
    L_len = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lx /= L_len
    Ly /= L_len
    Lz /= L_len

    # Diffuse shading (Lambertian)
    diffuse = Nx * Lx + Ny * Ly + Nz * Lz
    diffuse = torch.clamp(diffuse, 0, 1)

    # Specular highlights (Blinn-Phong) for velvety sheen
    # View vector (looking straight down)
    Vx, Vy, Vz = 0.0, 0.0, 1.0
    # Halfway vector
    Hx, Hy, Hz = Lx + Vx, Ly + Vy, Lz + Vz
    H_len = np.sqrt(Hx**2 + Hy**2 + Hz**2)
    Hx /= H_len; Hy /= H_len; Hz /= H_len
    
    spec_angle = torch.clamp(Nx * Hx + Ny * Hy + Nz * Hz, 0, 1)
    specular = torch.pow(spec_angle, 15.0) * 0.3 # Low gloss for velvet petals

    # Ambient Occlusion
    # Deeper crevices (lower H) get darker
    h_min, h_max = H[H > -5].min(), H.max()
    ao = torch.clamp((H - h_min) / (h_max - h_min), 0, 1)
    ao = 0.2 + 0.8 * ao # Base ambient + scaled
    
    # Mask out the background
    flower_mask = (H > -5.0).float()

    # Final composite
    final_img = color_map * (diffuse * 0.8 + 0.2) * ao.unsqueeze(0)
    final_img += specular.unsqueeze(0)
    
    # Tone mapping (Photographic exposure)
    exposure = 1.5
    final_img = 1.0 - torch.exp(-final_img * exposure)
    
    # Mask to black background
    final_img *= flower_mask.unsqueeze(0)

    final_img = torch.clamp(final_img, 0, 1)
    img_np = final_img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(img_np, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200, facecolor='black')
    plt.close()

render_photoreal("scratch/photoreal_flower.png")
print("Rendered Photorealistic Flower.")
