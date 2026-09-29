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

def render_einstein(filename):
    res = 1000
    x = torch.linspace(-2.0, 2.0, res)
    y = torch.linspace(-2.0, 2.0, res)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    H = torch.zeros_like(X)

    # 1. Base Face (Sum of soft Gaussians)
    # Forehead
    H += 1.0 * torch.exp(-((X)**2 + (Y-0.4)**2)/0.3)
    # Cheeks
    H += 0.8 * torch.exp(-((X-0.35)**2 + (Y-0.0)**2)/0.2)
    H += 0.8 * torch.exp(-((X+0.35)**2 + (Y-0.0)**2)/0.2)
    # Chin
    H += 0.7 * torch.exp(-((X)**2 + (Y+0.4)**2)/0.15)
    # Jawline
    H += 0.5 * torch.exp(-((X-0.3)**2 + (Y+0.3)**2)/0.15)
    H += 0.5 * torch.exp(-((X+0.3)**2 + (Y+0.3)**2)/0.15)

    # 2. Nose
    H += 0.4 * torch.exp(-((X)**2 + (Y-0.1)**2)/0.08) # bridge
    H += 0.3 * torch.exp(-((X)**2 + (Y+0.15)**2)/0.03) # tip

    # 3. Deep Eye Sockets (Subtraction)
    socket_r = torch.exp(-((X-0.25)**2 + (Y-0.05)**2)/0.03)
    socket_l = torch.exp(-((X+0.25)**2 + (Y-0.05)**2)/0.03)
    H -= 0.6 * socket_r
    H -= 0.6 * socket_l
    
    # Tiny eyeballs inside the sockets
    H += 0.2 * torch.exp(-((X-0.25)**2 + (Y-0.05)**2)/0.005)
    H += 0.2 * torch.exp(-((X+0.25)**2 + (Y-0.05)**2)/0.005)

    # 4. Mustache
    mustache_area = torch.exp(-((X)**2 + (Y+0.25)**2)/0.1)
    strands = torch.cos(80 * X + 5 * fbm(X*10, Y*10, octaves=3))
    H += 0.3 * mustache_area * (1.0 + 0.3 * strands)

    # 5. Wrinkles (Forehead & Nasolabial folds)
    forehead_wrinkles = torch.cos(40 * Y) * torch.exp(-((X)**2 + (Y-0.5)**2)/0.1)
    H += 0.05 * forehead_wrinkles
    
    naso_r = torch.exp(-((X-0.2 - Y*0.3)**2 + (Y+0.1)**2)/0.05)
    naso_l = torch.exp(-((X+0.2 + Y*0.3)**2 + (Y+0.1)**2)/0.05)
    H += 0.1 * naso_r + 0.1 * naso_l

    # 6. Wild Hair (Massive chaotic Gaussians masked by FBM)
    hair_volume = torch.exp(-((X)**2 + (Y-0.2)**2)/1.5)
    face_volume = torch.exp(-((X)**2 + (Y)**2)/0.6)
    hair_volume = torch.clamp(hair_volume - face_volume, 0, 1)
    
    hair_noise = fbm(X*8, Y*8, octaves=5)
    theta = torch.atan2(Y-0.2, X)
    hair_strands = torch.cos(60 * theta + 5 * hair_noise)
    
    H += 1.2 * hair_volume * (1.0 + 0.2 * hair_strands + 0.3 * hair_noise)
    
    # 7. Coat
    coat_area = torch.sigmoid(-20 * (Y + 0.5 - 0.2*X**2)) # Bottom of the image
    coat_tex = fbm(X*30, Y*30, octaves=4)
    H = torch.where(coat_area > 0.5, 0.5 + 0.1 * coat_tex, H)

    # --- Lighting & Rendering ---
    dHdx = torch.zeros_like(H)
    dHdy = torch.zeros_like(H)
    dHdx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / 2.0
    dHdy[1:-1, :] = (H[2:, :] - H[:-2, :]) / 2.0
    
    bump_scale = 15.0
    Nx = -dHdx * bump_scale
    Ny = -dHdy * bump_scale
    Nz = torch.ones_like(H)
    
    N_len = torch.sqrt(Nx**2 + Ny**2 + Nz**2)
    Nx /= N_len; Ny /= N_len; Nz /= N_len

    ao = torch.ones_like(H)
    for dx, dy in [(4,4), (-4,-4), (4,-4), (-4,4), (8,0), (0,8), (-8,0), (0,-8)]:
        H_shift = torch.roll(H, shifts=(dx, dy), dims=(0,1))
        occlusion = torch.clamp(H_shift - H, 0, 0.5)
        ao -= occlusion * 0.15
    ao = torch.clamp(ao, 0.2, 1.0)
    
    Lx, Ly, Lz = -0.8, 0.6, 0.4
    L_len = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    Lx /= L_len; Ly /= L_len; Lz /= L_len

    diffuse = Nx * Lx + Ny * Ly + Nz * Lz
    diffuse = torch.clamp(diffuse, 0, 1)

    Vx, Vy, Vz = 0.0, 0.0, 1.0
    Hx, Hy, Hz = Lx + Vx, Ly + Vy, Lz + Vz
    H_len = np.sqrt(Hx**2 + Hy**2 + Hz**2)
    Hx /= H_len; Hy /= H_len; Hz /= H_len
    
    spec_angle = torch.clamp(Nx * Hx + Ny * Hy + Nz * Hz, 0, 1)
    specular = torch.pow(spec_angle, 25.0) * 0.2

    final_img = (diffuse * 0.8 + 0.2) * ao
    final_img += specular
    
    exposure = 2.5
    final_img = 1.0 - torch.exp(-final_img * exposure)
    final_img = torch.clamp(final_img, 0, 1)
    
    final_img_rgb = final_img.unsqueeze(0).repeat(3, 1, 1)
    img_np = final_img_rgb.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(img_np, origin='lower')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200, facecolor='black')
    plt.close()

if __name__ == "__main__":
    render_einstein("scratch/einstein_yeganeh.png")
    print("Rendered Einstein")
