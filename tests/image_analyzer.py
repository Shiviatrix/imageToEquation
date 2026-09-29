import cv2
import numpy as np
from typing import Dict, List, Tuple

class ImageAnalyzer:
    """Automated image analysis pipeline."""
    
    def __init__(self, image: np.ndarray):
        """
        Args:
            image: [height, width, 3] uint8 RGB image
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        self.edges = None
        self.color_palette = None
        self.fft_spectrum = None
        self.dominant_frequencies = None
        self.gradient_magnitude = None
        self.symmetry_score = None
        self.centroids = None
    
    def extract_edges(self, blur_kernel: int = 5, canny_low: int = 50, canny_high: int = 150):
        blurred = cv2.GaussianBlur(self.gray, (blur_kernel, blur_kernel), 1.0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        self.edges = edges
        edge_density = np.sum(edges > 0) / (self.height * self.width)
        return edges, edge_density
    
    def extract_color_palette(self, num_colors: int = 4, num_samples: int = 10000):
        pixels = self.image.reshape(-1, 3).astype(np.float32)
        
        # Sample for speed if too large
        if pixels.shape[0] > num_samples:
            sample_indices = np.random.choice(pixels.shape[0], num_samples, replace=False)
            sampled_pixels = pixels[sample_indices]
        else:
            sampled_pixels = pixels
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(sampled_pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(-counts)
        
        palette = centers[sorted_idx].astype(np.uint8)
        weights = counts[sorted_idx] / np.sum(counts)
        
        self.color_palette = palette
        return palette, weights
    
    def extract_frequency_spectrum(self, blur_sigma: float = 2.0):
        blurred = cv2.GaussianBlur(self.gray, (0, 0), blur_sigma)
        fft = np.fft.fft2(blurred)
        spectrum = np.abs(fft)
        spectrum_log = np.log1p(spectrum)
        
        spectrum_centered = np.fft.fftshift(spectrum_log)
        
        cy, cx = self.height // 2, self.width // 2
        
        # Mask out DC component
        mask_radius = 5
        y, x = np.ogrid[-cy:self.height-cy, -cx:self.width-cx]
        mask = x*x + y*y <= mask_radius*mask_radius
        spectrum_centered[mask] = 0
        
        flat = spectrum_centered.flatten()
        top_indices = np.argsort(-flat)[:10]
        top_freqs = []
        
        for idx in top_indices:
            iy, ix = np.unravel_index(idx, spectrum_centered.shape)
            freq_y = iy - cy
            freq_x = ix - cx
            magnitude = flat[idx]
            top_freqs.append((freq_x, freq_y, magnitude))
            
        self.fft_spectrum = spectrum_centered
        self.dominant_frequencies = top_freqs
        
        return spectrum_centered, top_freqs
    
    def compute_gradient_magnitude(self):
        grad_x = cv2.Sobel(self.gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(self.gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_density = np.mean(magnitude) / 255.0
        self.gradient_magnitude = magnitude
        
        return magnitude, texture_density
    
    def detect_symmetry(self):
        radial_scores = []
        for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = cv2.rotate(self.gray, angle)
            # Ensure dimensions match before corrcoef, skip if not square and rotated 90
            if rotated.shape == self.gray.shape:
                score = np.corrcoef(self.gray.flatten(), rotated.flatten())[0, 1]
                radial_scores.append(score)
                
        radial_symmetry = np.mean(radial_scores) if radial_scores else 0.0
        
        mirror_scores = []
        for axis in [0, 1]: # 0 is vertical flip (around x-axis), 1 is horizontal
            flipped = cv2.flip(self.gray, axis)
            score = np.corrcoef(self.gray.flatten(), flipped.flatten())[0, 1]
            mirror_scores.append(score)
            
        max_radial = max(radial_scores) if radial_scores else 0.0
        max_mirror = max(mirror_scores) if mirror_scores else 0.0
        
        if max_radial > 0.6:
            primary = 'radial'
            sym_score = max_radial
        elif max_mirror > 0.6:
            primary = 'mirror'
            sym_score = max_mirror
        else:
            primary = 'none'
            sym_score = max(max_radial, max_mirror)
            
        self.symmetry_score = sym_score
        return sym_score, primary

    def estimate_object_count(self) -> int:
        if self.edges is None:
            self.extract_edges()
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
        
        min_area = 100
        significant_blobs = []
        valid_centroids = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                significant_blobs.append(i)
                valid_centroids.append(centroids[i])
                
        self.centroids = np.array(valid_centroids)
        object_count = max(1, min(100, len(significant_blobs)))
        return object_count
        
    def detect_instancing_pattern(self) -> str:
        if self.centroids is None or len(self.centroids) < 3:
            return 'random'
            
        distances = np.linalg.norm(self.centroids[:, np.newaxis] - self.centroids[np.newaxis, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        cv = np.std(min_distances) / (np.mean(min_distances) + 1e-6)
        
        if cv < 0.2:
            return 'grid'
        elif cv < 0.4:
            return 'spiral'
        else:
            return 'random'

    def summarize(self) -> Dict:
        edges, edge_density = self.extract_edges()
        palette, palette_weights = self.extract_color_palette()
        spectrum, top_freqs = self.extract_frequency_spectrum()
        gradients, texture_density = self.compute_gradient_magnitude()
        symmetry_score, symmetry_type = self.detect_symmetry()
        object_count = self.estimate_object_count()
        pattern = self.detect_instancing_pattern()
        
        return {
            'edge_density': edge_density,
            'texture_density': texture_density,
            'color_palette': palette,
            'dominant_frequencies': top_freqs,
            'symmetry_score': symmetry_score,
            'symmetry_type': symmetry_type,
            'object_count': object_count,
            'instancing_pattern': pattern
        }
