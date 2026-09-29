import torch
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)

def get_mgrid(sidelen, dim=2):
    # Generates a flattened grid of (x,y) coordinates in the range [-1, 1]
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def train_siren():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load target image
    img_path = "/Users/babayaga/.gemini/antigravity-ide/brain/90533dfc-b6d3-4690-bd2b-b753925d4ef6/original_einstein.jpg"
    img = Image.open(img_path).convert('RGB')
    
    # Resize to 256x256 for fast fitting
    res = 256
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
    
    # The image is loaded as [H, W, 3]. To match meshgrid indexing 'ij', we transpose it so [X, Y] aligns correctly.
    img_tensor = img_tensor.permute(1, 0, 2)
    
    # Target pixel values
    ground_truth = img_tensor.reshape(-1, 3).to(device)
    
    # Pixel coordinates mapped from -1 to 1
    coords = get_mgrid(res, 2).to(device)
    
    # 2. Initialize SIREN Mathematical Equation
    siren = Siren(in_features=2, out_features=3, hidden_features=256, hidden_layers=4, outermost_linear=True).to(device)
    
    optim = torch.optim.Adam(lr=1e-4, params=siren.parameters())
    
    # 3. Optimization Loop
    print("Beginning SIREN Reverse Yeganeh optimization...")
    steps = 4001
    for step in range(steps):
        model_output = siren(coords)
        loss = ((model_output - ground_truth)**2).mean()
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if step % 500 == 0:
            print(f"Step {step:04d}, Loss: {loss.item():.6f}")

    print("Mathematical fitting complete! Generating infinite resolution render...")
    
    # 4. Render at High Resolution (demonstrating infinite math scaling)
    high_res = 1000
    coords_high = get_mgrid(high_res, 2).to(device)
    
    with torch.no_grad():
        final_img = siren(coords_high)
        final_img = torch.clamp(final_img, 0, 1)
        final_img = final_img.view(high_res, high_res, 3).cpu().numpy()
        
    # Rotate back to Image format [H, W, 3]
    final_img = np.transpose(final_img, (1, 0, 2))
    
    plt.imsave("scratch/siren_einstein_render.png", final_img)
    print("Render saved to scratch/siren_einstein_render.png")

if __name__ == '__main__':
    train_siren()
