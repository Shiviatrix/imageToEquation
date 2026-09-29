import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_symbolic import ImageToAST, YeganehVocabulary, program_to_source, validate_program


def _small_circle(vocabulary: YeganehVocabulary):
    return [
        "SCENE",
        vocabulary.quantize(0.0), vocabulary.quantize(0.0), vocabulary.quantize(0.0),
        vocabulary.quantize(0.2), vocabulary.quantize(0.6), vocabulary.quantize(0.9),
        vocabulary.quantize(0.06),
        "sub", "pow2", "sub", "x", vocabulary.quantize(0.5), vocabulary.quantize(0.1),
        "<eos>",
    ]


def test_prefix_program_is_safe_and_lowers_to_evaluator_source():
    vocabulary = YeganehVocabulary()
    program = _small_circle(vocabulary)

    validate_program(program, vocabulary)
    source = program_to_source(program, vocabulary)

    assert source.startswith("over(rgb(")
    assert "sdfmask" in source
    with pytest.raises(ValueError):
        validate_program(["sin", "x", "<eos>"], vocabulary)


def test_compact_image_to_ast_teacher_forcing_shape():
    vocabulary = YeganehVocabulary()
    model = ImageToAST(vocabulary=vocabulary, d_model=96, layers=2, heads=4, max_length=32)
    inputs = torch.tensor([[vocabulary.bos_id, vocabulary.token_to_id["SCENE"]]])

    logits = model(torch.rand(1, 3, 64, 64), inputs)

    assert logits.shape == (1, 2, len(vocabulary))
