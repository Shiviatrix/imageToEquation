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

    noise_warp1 = fbm(X*2, Y*2, octaves=4)
    noise_warp2 = fbm(X*2+10, Y*2-10, octaves=4)
    X_w = X + 0.08 * noise_warp1
    Y_w = Y + 0.08 * noise_warp2

    r = torch.sqrt(X_w**2 + Y_w**2) + 1e-5
    theta = torch.atan2(Y_w, X_w)

    H = torch.full_like(r, -10.0)
    # Track Layer ID for crisp edges
    LayerID = torch.full_like(r, -1.0)
    color_map = torch.zeros((3, res, res))

    num_petals = 40
    golden_angle = 137.5 * np.pi / 180.0
    
    for i in range(num_petals):
        p_theta = i * golden_angle
        p_radius = 0.05 + (i / num_petals)**1.2 * 0.9 
        p_width = 0.22 + 0.25 * (i / num_petals) # thinner for more separation
        p_length = 0.3 + 0.4 * (i / num_petals)
        
        d_theta = (theta - p_theta) % (2 * np.pi)
        d_theta = torch.where(d_theta > np.pi, 2 * np.pi - d_theta, d_theta)
        
        # Parabolic profiles
        width_profile = 1.0 - (d_theta / p_width)**2
        length_profile = 1.0 - ((r - p_radius) / p_length)**2
        
        petal_mask = (width_profile > 0) & (length_profile > 0)
        
        # Huge Z-separation
        layer_z = i * 0.15
        droop = -0.4 * (r - 0.2)**2
        petal_h = layer_z + 0.3 * width_profile * length_profile + droop
        
        # Wrinkles
        wrinkles = 0.04 * fbm(X*15 + i, Y*15 - i, octaves=3)
        petal_h += wrinkles * torch.clamp(width_profile, 0, 1) * torch.clamp(length_profile, 0, 1)
        
        update_mask = (petal_h > H) & petal_mask
        
        H = torch.where(update_mask, petal_h, H)
        LayerID = torch.where(update_mask, torch.full_like(r, float(i)), LayerID)
        
        # Gradient color
        center_color = torch.tensor([0.4, 0.0, 0.05]).view(3,1,1)
        edge_color = torch.tensor([0.9, 0.1, 0.2]).view(3,1,1)
        blend = torch.clamp((r / 1.5), 0, 1)
        p_color = center_color * (1 - blend) + edge_color * blend
        
        veins = torch.clamp(torch.cos(60 * r + 15 * torch.cos(10 * theta)), 0, 1)
        p_color -= 0.05 * veins
        
        for c in range(3):
            color_map[c] = torch.where(update_mask, p_color[c] * torch.ones_like(r), color_map[c])

    # Compute Normals
    dHdx = torch.zeros_like(H)
    dHdy = torch.zeros_like(H)
    dHdx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / 2.0
    dHdy[1:-1, :] = (H[2:, :] - H[:-2, :]) / 2.0
    
    # Sharp Normal Edges
    dLayerX = torch.zeros_like(LayerID)
    dLayerY = torch.zeros_like(LayerID)
    dLayerX[:, 1:-1] = torch.abs(LayerID[:, 2:] - LayerID[:, :-2])
    dLayerY[1:-1, :] = torch.abs(LayerID[2:, :] - LayerID[:-2, :])
    
    bump_scale = 25.0
    Nx = -dHdx * bump_scale
    Ny = -dHdy * bump_scale
    Nz = torch.ones_like(H)
    
    N_len = torch.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nx /= N_len; Ny /= N_len; Nz /= N_len

    # SSAO (Screen Space Ambient Occlusion) Drop Shadows
    ao = torch.ones_like(H)
    # 8-neighbor shadow sampling
    for dx in [-4, 0, 4]:
        for dy in [-4, 0, 4]:
            if dx == 0 and dy == 0: continue
            H_shift = torch.roll(H, shifts=(dx, dy), dims=(0,1))
            # If the neighbor pixel is physically higher than me, it casts a shadow on me!
            occlusion = torch.clamp(H_shift - H, 0, 0.5)
            ao -= occlusion * 0.25
    ao = torch.clamp(ao, 0.1, 1.0)
    
    # Global depth darkening
    h_min, h_max = H[H > -5].min(), H.max()
    depth_ao = torch.clamp((H - h_min) / (h_max - h_min), 0, 1)
    ao *= (0.2 + 0.8 * depth_ao)

    # Lighting
    Lx, Ly, Lz = -0.5, 0.5, 0.8
    L_len = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lx /= L_len; Ly /= L_len; Lz /= L_len

    diffuse = Nx * Lx + Ny * Ly + Nz * Lz
    diffuse = torch.clamp(diffuse, 0, 1)

    Vx, Vy, Vz = 0.0, 0.0, 1.0
    Hx, Hy, Hz = Lx + Vx, Ly + Vy, Lz + Vz
    H_len = np.sqrt(Hx**2 + Hy**2 + Hz**2)
    Hx /= H_len; Hy /= H_len; Hz /= H_len
    
    spec_angle = torch.clamp(Nx * Hx + Ny * Hy + Nz * Hz, 0, 1)
    specular = torch.pow(spec_angle, 15.0) * 0.2

    flower_mask = (H > -5.0).float()

    final_img = color_map * (diffuse * 0.7 + 0.3) * ao.unsqueeze(0)
    final_img += specular.unsqueeze(0)
    
    # Contact shadow right on the edge seam of the petals
    edge_mask = torch.clamp(dLayerX + dLayerY, 0, 1)
    final_img *= (1.0 - 0.5 * edge_mask).unsqueeze(0)
    
    exposure = 1.5
    final_img = 1.0 - torch.exp(-final_img * exposure)
    
    final_img *= flower_mask.unsqueeze(0)
    final_img = torch.clamp(final_img, 0, 1)
    img_np = final_img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(img_np, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200, facecolor='black')
    plt.close()

render_photoreal("scratch/photoreal_flower_v2.png")
print("Rendered Photorealistic Flower V2.")
