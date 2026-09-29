import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def render_bird(filename, eps=0.0):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    k_vals = np.arange(1, 9831)
    
    # X(k)
    term1_x = (np.sin(k_vals * np.pi / 20000))**12
    term2_x = 0.5 * (np.cos(31 * np.pi * k_vals / 10000))**16 * np.sin(np.pi * k_vals / 10000)
    term3_x = (1/6) * (np.sin(31 * np.pi * k_vals / 10000))**20
    term4_x = k_vals / 20000
    term5_x = (np.cos(31 * np.pi * k_vals / 10000))**4 * np.sin(0.5 * ((k_vals - 10000)/10000)**2 - eps/2.0)
    X = term1_x * (term2_x + term3_x) + term4_x + term5_x
    
    # Y(k)
    term1_y = -1.5 * (np.cos(31 * np.pi * k_vals / 10000))**6 * np.cos(0.5 * ((k_vals - 10000)/10000)**2 - eps)
    term2_y = 3 + (np.sin(np.pi * k_vals / 20000) * np.sin(3 * np.pi * k_vals / 20000))**9
    term3_y = 0.75 * (np.cos(31 * np.pi * (k_vals - 10000) / 10000))**10 * (np.cos(9 * np.pi * (k_vals - 10000) / 10000))**10 * (np.cos(31 * np.pi * (k_vals - 10000) / 10000))**14
    term4_y = (1/20) * ((k_vals - 11500)/14000)**2
    Y = term1_y * term2_y + term3_y + term4_y
    
    # R(k)
    term1_r = (np.sin(np.pi * k_vals / 20000))**16
    term2_r = 0.25 * (np.cos(31 * np.pi * k_vals / 10000 + 25/15))**20 + 0.75 * (np.cos(31 * np.pi * k_vals / 10000))**2
    term3_r = 0.15 * (1.5 - (np.cos(62 * np.pi * k_vals / 10000))**2)
    R = term1_r * term2_r + term3_r
    
    # Draw circles
    for i in range(len(k_vals)):
        # Very small circles
        circ = Circle((X[i], Y[i]), R[i] * 0.015, color='black', alpha=0.5, linewidth=0)
        ax.add_patch(circ)
        
    ax.autoscale()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

# Render baseline bird
render_bird("scratch/bird_baseline.png", eps=0.0)

# Mutate epsilon (wings shift)
render_bird("scratch/bird_mutated_1.png", eps=3.14)
render_bird("scratch/bird_mutated_2.png", eps=-3.14)

print("Rendered Bird tests.")
