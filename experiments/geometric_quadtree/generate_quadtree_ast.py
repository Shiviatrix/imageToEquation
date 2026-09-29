import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from image_to_equation.evaluator import GridEvaluator

class QuadTreeCell:
    """
    A geometric node in the Binary Space Partitioning (QuadTree) algorithm.
    Recursively subdivides an image patch if its pixel variance exceeds a threshold,
    bottoming out at a solid color geometric box constraint.
    """
    def __init__(self, target_image: np.ndarray, x: int, y: int, size: int, depth: int, max_depth: int, var_thresh: float):
        self.target_image = target_image
        self.x = x
        self.y = y
        self.size = size
        self.depth = depth
        self.max_depth = max_depth
        self.var_thresh = var_thresh
        
        self.children: List['QuadTreeCell'] = []
        self.is_leaf = False
        self.brightness = 0.0
        
        self._build_tree()
        
    def _build_tree(self):
        """Recursively builds the quadtree based on image variance."""
        
        # Extract the current spatial patch
        patch = self.target_image[self.y : self.y + self.size, self.x : self.x + self.size]
        
        if patch.size == 0:
            self.is_leaf = True
            self.brightness = 0.0
            return
            
        variance = np.var(patch)
        self.brightness = float(np.mean(patch))
        
        # Stop partitioning if we hit max depth, or if the variance is low (the patch is a solid color)
        if self.depth >= self.max_depth or variance < self.var_thresh or self.size <= 1:
            self.is_leaf = True
        else:
            half = self.size // 2
            self.children.append(QuadTreeCell(self.target_image, self.x, self.y, half, self.depth + 1, self.max_depth, self.var_thresh))
            self.children.append(QuadTreeCell(self.target_image, self.x + half, self.y, half, self.depth + 1, self.max_depth, self.var_thresh))
            self.children.append(QuadTreeCell(self.target_image, self.x, self.y + half, half, self.depth + 1, self.max_depth, self.var_thresh))
            self.children.append(QuadTreeCell(self.target_image, self.x + half, self.y + half, half, self.depth + 1, self.max_depth, self.var_thresh))
            
    def extract_geometric_bounds(self, resolution: int) -> List[tuple]:
        """Returns the list of bounding boxes (cx, cy, w, h, brightness) for all leaf nodes."""
        leaves = []
        if self.is_leaf:
            # Map pixel coordinates to the continuous [-1, 1] mathematical space
            cx = (self.x + self.size / 2.0) / resolution
            cy = (self.y + self.size / 2.0) / resolution
            w = self.size / resolution
            h = self.size / resolution
            
            # Optimization: Skip purely black boxes to save AST characters
            if self.brightness > 0.02:
                leaves.append((cx, cy, w, h, self.brightness))
        else:
            for child in self.children:
                leaves.extend(child.extract_geometric_bounds(resolution))
        return leaves

def generate_quadtree_ast():
    """
    Reads a target image and compiles it into a colossal Yeganeh AST 
    using spatial partitioning (QuadTree).
    """
    print("Loading image for QuadTree decomposition...")
    
    # We resolve the absolute path so this script can be run from anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "assets", "einstein_matched.jpg")
    
    img = Image.open(img_path).convert('L')
    res = 256 # Base resolution
    img = img.resize((res, res), Image.Resampling.LANCZOS)
    target_image = np.array(img).astype(np.float32) / 255.0
    
    # Create the procedural quadtree
    qt = QuadTreeCell(target_image, 0, 0, res, depth=0, max_depth=7, var_thresh=0.005)
    geometric_leaves = qt.extract_geometric_bounds(res)
    
    print(f"QuadTree partitioned the image into {len(geometric_leaves)} geometric bounds.")
    
    # Cap the bounds to prevent Python Recursion Limits during evaluation
    if len(geometric_leaves) > 4000:
        print("Optimizing quadtree bounds for AST compilation limits...")
        qt = QuadTreeCell(target_image, 0, 0, res, depth=0, max_depth=8, var_thresh=0.001)
        geometric_leaves = qt.extract_geometric_bounds(res)
        print(f"Max Resolution Adjusted: {len(geometric_leaves)} boxes.")
        
    print("Compiling AST string...")
    ast_terms = []
    for (cx, cy, w, h, brightness) in geometric_leaves:
        # sdfmask acts as a continuous boolean intersection returning 1 inside the box and 0 outside
        term = f"{brightness:.3f}*sdfmask(box(x, y, {cx:.4f}, {cy:.4f}, {w/2:.4f}, {h/2:.4f}), 0.001)"
        ast_terms.append(term)
        
    poly_expr = " + ".join(ast_terms)
    
    full_ast = f"""
    let(brightness, clamp({poly_expr}, 0.0, 1.0),
      rgb(brightness, brightness, brightness)
    )
    """
    
    full_ast = full_ast.replace("\n", "").replace(" ", "").strip()
    print(f"AST Length: {len(full_ast)}")
    
    out_txt = os.path.join(base_dir, "outputs", "einstein_quadtree_ast.txt")
    with open(out_txt, "w") as f:
        f.write(full_ast)
        
    print("Evaluating Procedural Image...")
    evaluator = GridEvaluator()
    try:
        # Prevent Python's C stack from overflowing on deep ASTs
        sys.setrecursionlimit(50000)
        img_rgb = evaluator.evaluate(full_ast, 800, 800, render_mode='linear')
        out_png = os.path.join(base_dir, "outputs", "rendered_images", "einstein_quadtree.png")
        plt.imsave(out_png, img_rgb.numpy())
        print(f"Success! Rendered to {out_png}")
    except Exception as e:
        print(f"Error evaluating AST: {e}")

if __name__ == '__main__':
    generate_quadtree_ast()
