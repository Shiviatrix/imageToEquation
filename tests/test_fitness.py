import pytest
import sys
import os
import torch

# Add parent dir to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gd_optimizer import enhanced_fitness
from evaluator import GridEvaluator

def test_enhanced_fitness_structured():
    evaluator = GridEvaluator()
    # Create fake target tensor [C, H, W]
    target = torch.ones(3, 64, 64) * 128.0
    
    eq_str = "cos(10*x)"
    # We expect a negative scalar tensor out
    loss = enhanced_fitness(eq_str, target, 'structured_art', evaluator)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() < 0

def test_enhanced_fitness_interference():
    evaluator = GridEvaluator()
    target = torch.ones(3, 64, 64) * 128.0
    eq_str = "cos(10*x)"
    loss = enhanced_fitness(eq_str, target, 'interference', evaluator)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() < 0

def test_enhanced_fitness_stochastic():
    evaluator = GridEvaluator()
    target = torch.ones(3, 64, 64) * 128.0
    eq_str = "cos(10*x)"
    loss = enhanced_fitness(eq_str, target, 'stochastic', evaluator)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() < 0
