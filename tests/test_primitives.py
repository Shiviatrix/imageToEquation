import pytest
import sys
import os

# Add parent dir to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genetic_algo import InterferencePattern, PerlinNoisePrimitive, ProceduralTerrain
from evaluator import GridEvaluator

def test_interference_pattern_compiles():
    params = {
        'num_waves': 2,
        'freq_0': 1.0, 'angle_0': 0.0, 'amp_0': 1.0, 'phase_0': 0.0,
        'freq_1': 2.0, 'angle_1': 1.5, 'amp_1': 0.5, 'phase_1': 0.5
    }
    prim = InterferencePattern(params)
    eq_str = prim.generate_string('x', 'y')
    
    # Must compile and evaluate without errors
    evaluator = GridEvaluator()
    tensor = evaluator.evaluate(eq_str, 64, 64)
    assert tensor is not None
    assert tensor.shape == (64, 64, 3)

def test_perlin_noise_compiles():
    params = {
        'octaves': 3,
        'lacunarity': 2.0,
        'persistence': 0.5,
        'scale': 1.0
    }
    prim = PerlinNoisePrimitive(params)
    eq_str = prim.generate_string('x', 'y')
    
    evaluator = GridEvaluator()
    tensor = evaluator.evaluate(eq_str, 64, 64)
    assert tensor is not None
    assert tensor.shape == (64, 64, 3)

def test_procedural_terrain_compiles():
    params = {
        'base_freq': 2.0,
        'roughness': 0.5,
        'warp_strength': 0.2
    }
    prim = ProceduralTerrain(params)
    eq_str = prim.generate_string('x', 'y')
    
    evaluator = GridEvaluator()
    tensor = evaluator.evaluate(eq_str, 64, 64)
    assert tensor is not None
    assert tensor.shape == (64, 64, 3)
