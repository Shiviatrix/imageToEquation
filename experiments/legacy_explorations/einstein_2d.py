import torch
import numpy as np
import matplotlib.pyplot as plt

def render_einstein():
    res = 800
    x = torch.linspace(-1, 1, res)
    y = torch.linspace(1, -1, res) # Image coordinates (top is +1, bottom is -1)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Base image: Dark gray background
    Img = torch.full_like(X, 0.1)

    def smoothstep(edge0, edge1, x):
        t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # 1. Hair
    R = torch.sqrt(X**2 + (Y-0.1)**2)
    Theta = torch.atan2(Y-0.1, X)
    
    # High frequency noise for hair strands
    noise1 = torch.sin(40*Theta) * torch.cos(30*R)
    noise2 = torch.sin(20*X) * torch.cos(20*Y)
    
    hair_d = R + 0.15*noise1 + 0.05*noise2
    hair_mask = smoothstep(0.85, 0.6, hair_d)
    
    hair_tex = 0.4 + 0.3 * torch.sin(100*Theta + 10*torch.sin(15*R))
    # Make it brighter near the top to simulate lighting
    hair_tex += 0.2 * smoothstep(0.0, 0.8, Y)
    
    Img = torch.where(hair_mask > 0, torch.lerp(Img, hair_tex, hair_mask), Img)

    # 2. Face Shape
    face_w = 0.42 + 0.12 * Y
    face_d = torch.sqrt((X/face_w)**2 + (Y/0.55)**2)
    face_mask = smoothstep(1.0, 0.95, face_d)
    
    face_color = torch.full_like(X, 0.65)
    
    # Face shading (Ambient Occlusion & Form shading)
    face_color -= 0.35 * smoothstep(0.6, 1.0, face_d) # Darker edges
    face_color += 0.15 * smoothstep(0.0, 0.8, Y)      # Lighter top (light source)
    
    # 3. Eyes
    eye_dx = torch.abs(X) - 0.18
    eye_dy = Y - 0.05
    
    # Dark Eye Sockets
    socket_d = torch.sqrt((eye_dx/0.14)**2 + (eye_dy/0.09)**2)
    face_color -= 0.4 * smoothstep(1.0, 0.3, socket_d)
    
    # Eyeball (Sclera)
    eye_d = torch.sqrt((eye_dx/0.07)**2 + (eye_dy/0.035)**2)
    eye_mask = smoothstep(1.0, 0.8, eye_d)
    face_color = torch.where(eye_mask > 0, torch.lerp(face_color, torch.full_like(X, 0.8), eye_mask), face_color)
    
    # Iris / Pupil
    iris_d = torch.sqrt((eye_dx/0.03)**2 + (eye_dy/0.03)**2)
    iris_mask = smoothstep(1.0, 0.8, iris_d)
    face_color = torch.where(iris_mask > 0, torch.lerp(face_color, torch.full_like(X, 0.15), iris_mask), face_color)
    
    # Eye Bags
    bag_d = torch.sqrt((eye_dx/0.12)**2 + ((eye_dy+0.06)/0.04)**2)
    face_color -= 0.15 * smoothstep(1.0, 0.5, bag_d)

    # 4. Nose
    nose_dy = Y + 0.1
    # Bridge
    bridge = smoothstep(0.1, 0.0, torch.abs(X)) * smoothstep(0.2, -0.2, nose_dy)
    face_color += 0.15 * bridge
    
    # Tip
    tip_d = torch.sqrt((X/0.09)**2 + ((Y+0.2)/0.06)**2)
    face_color += 0.15 * smoothstep(1.0, 0.0, tip_d)
    face_color -= 0.1 * smoothstep(0.5, 1.0, tip_d) # underside shadow
    
    # Nostrils
    nostril_d = torch.sqrt(((torch.abs(X)-0.05)/0.03)**2 + ((Y+0.22)/0.015)**2)
    face_color -= 0.5 * smoothstep(1.0, 0.5, nostril_d)

    # 5. Mustache
    must_dx = X
    must_dy = Y + 0.32
    must_d = torch.sqrt((must_dx/0.28)**2 + ((must_dy + 0.6*X**2)/0.1)**2)
    must_mask = smoothstep(1.0, 0.7, must_d)
    
    must_tex = 0.5 + 0.4 * torch.sin(180*X + 8*torch.sin(60*Y))
    # Mustache shadow
    face_color -= 0.2 * smoothstep(1.5, 0.5, must_d)
    face_color = torch.where(must_mask > 0, torch.lerp(face_color, must_tex, must_mask), face_color)

    # 6. Wrinkles & Texture
    # Forehead
    forehead_w = torch.sin(50*Y + 6*torch.sin(15*X))
    forehead_mask = smoothstep(0.1, 0.4, Y) * smoothstep(0.3, 0.0, torch.abs(X))
    face_color -= 0.08 * forehead_w * forehead_mask
    
    # Nasolabial folds (smile lines)
    smile_d = torch.sqrt(((torch.abs(X)-0.15 - 0.5*(Y+0.2)**2)/0.02)**2 + ((Y+0.25)/0.15)**2)
    face_color -= 0.15 * smoothstep(1.0, 0.0, smile_d)

    # Composite Face over Hair
    Img = torch.where(face_mask > 0, torch.lerp(Img, face_color, face_mask), Img)

    # 7. Coat & Collar
    collar_mask = smoothstep(0.02, -0.02, Y + 0.55 + 0.4*torch.abs(X))
    collar_color = torch.full_like(X, 0.8) # White shirt collar
    Img = torch.where(collar_mask > 0, torch.lerp(Img, collar_color, collar_mask), Img)
    
    coat_mask = smoothstep(0.05, -0.05, Y + 0.65 - 0.2*X**2)
    coat_color = 0.25 + 0.05 * torch.sin(120*X + 10*torch.sin(120*Y)) # Tweed texture
    Img = torch.where(coat_mask > 0, torch.lerp(Img, coat_color, coat_mask), Img)

    # Output
    Img = torch.clamp(Img, 0.0, 1.0)
    plt.imsave('scratch/einstein_2d.png', Img.numpy(), cmap='gray')
    print("Saved 2D procedural Einstein.")

if __name__ == '__main__':
    render_einstein()
