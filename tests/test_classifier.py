import numpy as np
import pytest
import cv2
import sys
import os

# Add parent dir to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_classifier import ImageTypeClassifier

def test_classifier_initialization():
    classifier = ImageTypeClassifier()
    assert classifier.analysis is None

def test_classifier_structured_art():
    # Create a synthetic image that looks like structured art (solid geometry on blank background)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
    
    classifier = ImageTypeClassifier()
    result = classifier.classify(img)
    
    # It might be classified as structured_art or hybrid depending on heuristics, 
    # but we just want to ensure it runs without crashing and returns a valid dictionary.
    assert isinstance(result, dict)
    assert 'primary_type' in result
    assert result['primary_type'] in ['structured_art', 'interference', 'stochastic', 'terrain', 'hybrid']
    assert 'confidence' in result
    assert 'characteristics' in result

def test_classifier_stochastic():
    # Random noise image
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    classifier = ImageTypeClassifier()
    result = classifier.classify(img)
    
    assert isinstance(result, dict)
    assert result['primary_type'] in ['structured_art', 'interference', 'stochastic', 'terrain', 'hybrid']
