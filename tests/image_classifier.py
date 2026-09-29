import cv2
import numpy as np
from typing import Dict, Tuple, List

class ImageTypeClassifier:
    """
    Classify images into structural categories using heuristics.
    
    Categories:
      - 'structured_art': Objects, instancing, clear geometry (Hamid-style)
      - 'interference': Caustics, wave patterns, ripples
      - 'stochastic': Clouds, smoke, turbulence, Perlin-like
      - 'terrain': Height fields, landscapes, mountains
      - 'hybrid': Mix of above
    """
    
    def __init__(self):
        self.analysis = None
    
    def classify(self, image: np.ndarray) -> Dict:
        # Run all analysis
        edge_density, edges = self._analyze_edges(image)
        texture_density, texture_freq = self._analyze_texture(image)
        symmetry_score, sym_type = self._analyze_symmetry(image)
        periodicity_score = self._analyze_periodicity(image)
        fractal_dimension = self._analyze_fractal_dimension(image)
        color_variance = self._analyze_color_variance(image)
        
        self.analysis = {
            'edge_density': edge_density,
            'texture_density': texture_density,
            'symmetry_score': symmetry_score,
            'symmetry_type': sym_type,
            'periodicity_score': periodicity_score,
            'fractal_dimension': fractal_dimension,
            'color_variance': color_variance,
        }
        
        return self._heuristic_classify()
    
    def _analyze_edges(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return edge_density, edges
    
    def _analyze_texture(self, image: np.ndarray) -> Tuple[float, Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        gradients = []
        for sigma in [1, 2, 4, 8]:
            grad = cv2.GaussianBlur(gray, (0, 0), sigma)
            grad_x = cv2.Sobel(grad, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(grad, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(grad_x**2 + grad_y**2)
            gradients.append(np.mean(mag))
        
        texture_density = np.mean(gradients) / 255.0
        texture_freq = self._compute_texture_frequency(gray)
        
        return texture_density, texture_freq
    
    def _compute_texture_frequency(self, gray: np.ndarray) -> Dict:
        fft = np.fft.fft2(gray)
        spectrum = np.abs(fft)
        spectrum_log = np.log1p(spectrum)
        
        cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
        Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        r_int = r.astype(int)
        
        # Prevent bincount errors if sizes mismatch, just basic radial avg
        spectrum_centered = np.fft.fftshift(spectrum_log)
        radial_sum = np.bincount(r_int.ravel(), spectrum_centered.ravel())
        radial_count = np.bincount(r_int.ravel())
        radial_count[radial_count == 0] = 1 # Avoid div by zero
        radial_spectrum = radial_sum / radial_count
        
        peak_freq = np.argmax(radial_spectrum[10:]) + 10 if len(radial_spectrum) > 10 else 1
        
        return {'peak_freq': peak_freq, 'radial_spectrum': radial_spectrum}
    
    def _analyze_symmetry(self, image: np.ndarray) -> Tuple[float, str]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        radial_scores = []
        for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = cv2.rotate(gray, angle)
            if rotated.shape == gray.shape:
                score = np.corrcoef(gray.flatten(), rotated.flatten())[0, 1]
                radial_scores.append(score)
                
        max_radial = max(radial_scores) if radial_scores else 0.0
        
        h_flip = cv2.flip(gray, 1)
        v_flip = cv2.flip(gray, 0)
        mirror_scores = [
            np.corrcoef(gray.flatten(), h_flip.flatten())[0, 1],
            np.corrcoef(gray.flatten(), v_flip.flatten())[0, 1],
        ]
        max_mirror = max(mirror_scores)
        
        if max_radial > 0.7:
            return max_radial, 'radial'
        elif max_mirror > 0.7:
            return max_mirror, 'mirror'
        else:
            return max(max_radial, max_mirror), 'none'
    
    def _analyze_periodicity(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fft2(gray)
        spectrum = np.abs(fft)
        spectrum_log = np.log1p(spectrum)
        
        spectrum_centered = np.fft.fftshift(spectrum_log)
        cy, cx = spectrum_centered.shape[0] // 2, spectrum_centered.shape[1] // 2
        
        # Mask DC
        mask_radius = 10
        y, x = np.ogrid[-cy:spectrum_centered.shape[0]-cy, -cx:spectrum_centered.shape[1]-cx]
        mask = x*x + y*y <= mask_radius*mask_radius
        spectrum_centered[mask] = 0
        
        total_energy = np.sum(spectrum_centered)
        top_10_energy = np.sum(np.sort(spectrum_centered.ravel())[-10:])
        
        periodicity = top_10_energy / (total_energy + 1e-10)
        return periodicity
    
    def _analyze_fractal_dimension(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            h, w = edges.shape
            # Ensure divisibility
            h = h - (h % scale)
            w = w - (w % scale)
            cropped_edges = edges[:h, :w]
            
            if h == 0 or w == 0:
                counts.append(1)
                continue
                
            boxes = cropped_edges.reshape(h//scale, scale, w//scale, scale)
            boxes = np.any(boxes, axis=(1, 3))
            count = np.sum(boxes)
            counts.append(count if count > 0 else 1)
            
        log_scales = np.log(scales)
        log_counts = np.log(counts)
        
        coeffs = np.polyfit(log_scales, log_counts, 1)
        fractal_dim = -coeffs[0]
        
        return np.clip(fractal_dim, 0.5, 2.0)
    
    def _analyze_color_variance(self, image: np.ndarray) -> float:
        pixels = image.reshape(-1, 3)
        entropies = []
        for ch in range(3):
            hist, _ = np.histogram(pixels[:, ch], bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            entropies.append(entropy)
            
        return np.mean(entropies) / 8.0
    
    def _heuristic_classify(self) -> Dict:
        a = self.analysis
        
        if a['periodicity_score'] > 0.0001 or (a['symmetry_score'] > 0.6 and a['fractal_dimension'] < 1.5):
            # Interference/Caustics tend to have regular, periodic wavelike structures and lower fractal dim
            if a['edge_density'] < 0.15:
                return {
                    'primary_type': 'interference',
                    'confidence': 0.85,
                    'secondary_types': [],
                    'characteristics': a,
                }
            
        if a['fractal_dimension'] > 1.6 or (a['texture_density'] > 0.12 and a['edge_density'] > 0.1):
            # Terrain/Heightmaps have high variance and structural texture
            return {
                'primary_type': 'terrain',
                'confidence': 0.75,
                'secondary_types': [],
                'characteristics': a,
            }
            
        if a['fractal_dimension'] > 1.45 and a['texture_density'] > 0.05:
            # Textures/Turbulence have high fractal dimension and texture
            return {
                'primary_type': 'stochastic',
                'confidence': 0.8,
                'secondary_types': [],
                'characteristics': a,
            }

        if a['periodicity_score'] < 0.0001 and a['texture_density'] < 0.2 and a['symmetry_score'] > 0.5:
            return {
                'primary_type': 'structured_art',
                'confidence': 0.9,
                'secondary_types': [],
                'characteristics': a,
            }
            
        return {
            'primary_type': 'hybrid',
            'confidence': 0.5,
            'secondary_types': ['structured_art', 'stochastic'],
            'characteristics': a,
        }
