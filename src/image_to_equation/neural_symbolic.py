"""Fast, grammar-constrained image-to-equation neural symbolic regression.

The model deliberately predicts a *small closed language*, rather than source
text.  A prefix program begins with ``SCENE`` (background RGB, foreground RGB,
edge softness) followed by one implicit 2-D field.  ``program_to_source`` is
the only route from tokens to executable source, so an inference result cannot
inject arbitrary Python or evaluator syntax.

This module is intentionally inference-oriented: the convolutional encoder
produces an 8x8 feature board for a 64x64 input, and a small Transformer emits
the AST one token at a time with grammar masks.  Its activation quantizers are
fake-quantization modules during training and can be replaced with PyTorch's
static int8 conversion for deployment after calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
LEAVES = ("x", "y", "t0", "t1", "t2")
# The synthetic curriculum retains a single smooth-union token as a codec
# composition operator.  It is not presented as a historical claim about
# Yeganeh; sin/cosine/powers remain the strict stylistic core.
UNARY = ("sin", "cos", "neg", "pow2", "dexp", "acos", "atan", "exp", "abs", "fmap")
BINARY = ("add", "sub", "mul", "div", "atan2", "let")
TERNARY = ("smin",)
QUINTARY = ("trignoise",)
SCENE = "SCENE"


class YeganehVocabulary:
    """Finite AST vocabulary with quantized literals and no free-form source.

    Constants cover ``[-2, 2]``.  This handles normalised coordinates and most
    trig amplitudes while retaining one-token parameters.  RGB header values
    are sampled from the legal ``[0, 1]`` subset by the synthetic generator.
    """

    def __init__(self, constant_bins: int = 65) -> None:
        if constant_bins < 3:
            raise ValueError("constant_bins must be at least 3")
        self.constant_bins = constant_bins
        tokens = list(SPECIAL_TOKENS) + [SCENE] + list(LEAVES) + list(UNARY)
        tokens += list(BINARY) + list(TERNARY) + list(QUINTARY)
        tokens += [f"C{index:02d}" for index in range(constant_bins)]
        self.tokens = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def token(self, index: int) -> str:
        return self.tokens[int(index)]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.token(int(index)) for index in ids]

    def is_constant(self, token: str) -> bool:
        return token.startswith("C") and token[1:].isdigit()

    def constant_value(self, token: str) -> float:
        if not self.is_constant(token):
            raise ValueError(f"not a quantized constant: {token}")
        return -2.0 + 4.0 * int(token[1:]) / (self.constant_bins - 1)

    def quantize(self, value: float) -> str:
        value = min(2.0, max(-2.0, float(value)))
        index = round((value + 2.0) * (self.constant_bins - 1) / 4.0)
        return f"C{index:02d}"

    def arity(self, token: str) -> int:
        if token == SCENE:
            return 8  # RGB background, RGB ink, softness, implicit field.
        if token in LEAVES or self.is_constant(token):
            return 0
        if token in UNARY:
            return 1
        if token in BINARY:
            return 2
        if token in TERNARY:
            return 3
        if token in QUINTARY:
            return 5
        raise ValueError(f"unknown AST token: {token}")


@dataclass
class GrammarState:
    """Incremental prefix grammar state used to mask decoder logits."""

    slots: int = 1
    header_constants_left: int = 0
    node_count: int = 0
    started: bool = False
    complete: bool = False

    def allowed_token_ids(self, vocabulary: YeganehVocabulary, max_nodes: int) -> list[int]:
        if self.complete:
            return [vocabulary.eos_id]
        if not self.started:
            return [vocabulary.token_to_id[SCENE]]
        if self.header_constants_left:
            return [
                vocabulary.token_to_id[token]
                for token in vocabulary.tokens
                if vocabulary.is_constant(token)
            ]
        if self.slots == 0:
            return [vocabulary.eos_id]
        allowed = list(LEAVES) + [token for token in vocabulary.tokens if vocabulary.is_constant(token)]
        if self.node_count < max_nodes:
            allowed += list(UNARY) + list(BINARY) + list(TERNARY) + list(QUINTARY)
        return [vocabulary.token_to_id[token] for token in allowed]

    def consume(self, token: str, vocabulary: YeganehVocabulary) -> None:
        if token == "<eos>":
            if self.started and self.slots == 0 and self.header_constants_left == 0:
                self.complete = True
                return
            raise ValueError("EOS before a complete program")
        if not self.started:
            if token != SCENE:
                raise ValueError("a program must begin with SCENE")
            self.started = True
            self.header_constants_left = 7
            self.slots = 1
            self.node_count = 1
            return
        if self.header_constants_left:
            if not vocabulary.is_constant(token):
                raise ValueError("SCENE header accepts only constants")
            self.header_constants_left -= 1
            return
        if self.slots <= 0:
            raise ValueError("program already complete")
        self.slots += vocabulary.arity(token) - 1
        self.node_count += 1


def validate_program(tokens: Sequence[str], vocabulary: YeganehVocabulary, max_nodes: int = 160) -> None:
    """Raise ``ValueError`` unless *tokens* form a complete safe prefix AST."""
    state = GrammarState()
    for token in tokens:
        if token in {"<pad>", "<bos>"}:
            raise ValueError("special token inside a program")
        allowed = state.allowed_token_ids(vocabulary, max_nodes)
        if vocabulary.token_to_id.get(token) not in allowed:
            raise ValueError(f"token {token!r} is invalid in the current grammar state")
        state.consume(token, vocabulary)
    if not state.complete:
        raise ValueError("incomplete program; append EOS only after all child slots close")


def _parse_field(tokens: Sequence[str], vocabulary: YeganehVocabulary, cursor: int = 0, state: dict | None = None) -> tuple[str, int]:
    if state is None:
        state = {"let_count": 0}
    token = tokens[cursor]
    arity = vocabulary.arity(token)
    if arity == 0:
        if token in LEAVES:
            return token, cursor + 1
        return f"{vocabulary.constant_value(token):.4g}", cursor + 1
    children: list[str] = []
    position = cursor + 1
    for _ in range(arity):
        child, position = _parse_field(tokens, vocabulary, position, state)
        children.append(child)
    if token == "add":
        return f"({children[0]}+{children[1]})", position
    if token == "sub":
        return f"({children[0]}-{children[1]})", position
    if token == "mul":
        return f"({children[0]}*{children[1]})", position
    if token == "div":
        return f"({children[0]}/({children[1]}+0.0001))", position
    if token == "atan2":
        return f"atan2({children[0]},{children[1]})", position
    if token == "let":
        var_name = f"t{state['let_count']}"
        state['let_count'] += 1
        return f"let({var_name},{children[0]},{children[1]})", position
    if token == "neg":
        return f"(-{children[0]})", position
    if token == "pow2":
        return f"(({children[0]})^2)", position
    if token == "dexp":
        return f"dexp_mask({children[0]},1000.0)", position
    if token in {"sin", "cos", "acos", "atan", "exp", "abs"}:
        return f"{token}({children[0]})", position
    if token == "fmap":
        return f"fmap({children[0]})", position
    if token == "smin":
        return f"smin({children[0]},{children[1]},abs({children[2]}))", position
    if token == "trignoise":
        # Convert the 5th continuous constant argument to an integer octave count 1..8
        try:
            # We emitted it as a constant string like '(-1.25)' if it was a constant,
            # or it might be a nested expression. The dataset guarantees it's a constant.
            val_str = children[4].strip("()")
            octaves = max(1, min(MAX_FBM_OCTAVES, int(round((float(val_str) + 2.0) * 1.75 + 1))))
        except Exception:
            octaves = 4
        return f"trignoise({children[0]},{children[1]},{children[2]},{children[3]},{octaves})", position
    raise ValueError(f"cannot emit unknown token {token}")


def program_to_source(tokens: Sequence[str], vocabulary: YeganehVocabulary) -> str:
    """Compile a valid token program to the existing evaluator's DSL."""
    validate_program(tokens, vocabulary)
    body = [token for token in tokens if token != "<eos>"]
    header = [vocabulary.constant_value(token) for token in body[1:8]]
    field, position = _parse_field(body, vocabulary, 8)
    if position != len(body):
        raise ValueError("unexpected trailing AST tokens")
    background = ",".join(f"{value:.4g}" for value in header[:3])
    foreground = ",".join(f"{value:.4g}" for value in header[3:6])
    softness = max(abs(header[6]), 1e-3)
    return f"over(rgb({background}),rgb({foreground}),sdfmask({field},{softness:.4g}))"


class FakeQuantize8(nn.Module):
    """Straight-through per-tensor symmetric int8 activation quantizer."""

    def forward(self, value: Tensor) -> Tensor:
        if not self.training:
            return value
        scale = value.detach().abs().amax().clamp_min(1e-6) / 127.0
        quantized = (value / scale).round().clamp(-127, 127) * scale
        return value + (quantized - value).detach()


class _DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.quantize = FakeQuantize8()

    def forward(self, value: Tensor) -> Tensor:
        return self.quantize(self.activation(self.norm(self.pointwise(self.depthwise(value)))))


class NNUEImageEncoder(nn.Module):
    """A compact CNN that returns an 8x8 spatial board plus global essence."""

    def __init__(self, d_model: int = 192, width: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.GELU(), FakeQuantize8(),
        )
        self.blocks = nn.Sequential(
            _DepthwiseBlock(width, width, 1),
            _DepthwiseBlock(width, width * 2, 2),
            _DepthwiseBlock(width * 2, width * 2, 1),
            _DepthwiseBlock(width * 2, d_model, 2),
        )
        self.memory_projection = nn.Conv2d(d_model, d_model, 1)
        self.global_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), FakeQuantize8())

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        features = self.memory_projection(self.blocks(self.stem(image)))
        memory = features.flatten(2).transpose(1, 2)  # B, 64, D for 64px inputs
        essence = self.global_projection(features.mean(dim=(-2, -1)))
        return memory, essence


class ImageToAST(nn.Module):
    """NNUE-style visual encoder and small grammar-constrained AST decoder."""

    def __init__(
        self,
        vocabulary: YeganehVocabulary | None = None,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 6,
        max_length: int = 192,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary or YeganehVocabulary()
        self.max_length = max_length
        self.encoder = NNUEImageEncoder(d_model=d_model)
        self.token_embedding = nn.Embedding(len(self.vocabulary), d_model)
        self.position_embedding = nn.Embedding(max_length + 1, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 3,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.memory_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, len(self.vocabulary), bias=False)

    def forward(self, image: Tensor, decoder_tokens: Tensor) -> Tensor:
        """Teacher-forced logits for decoder tokens of shape ``[B, length]``."""
        if decoder_tokens.ndim != 2 or decoder_tokens.shape[1] > self.max_length:
            raise ValueError(f"decoder tokens must be [batch, <= {self.max_length}]")
        memory, essence = self.encoder(image)
        memory = self.memory_norm(memory + essence.unsqueeze(1))
        positions = torch.arange(decoder_tokens.shape[1], device=image.device)
        target = self.token_embedding(decoder_tokens) + self.position_embedding(positions).unsqueeze(0)
        causal = torch.full((target.shape[1], target.shape[1]), float("-inf"), device=image.device)
        causal = torch.triu(causal, diagonal=1)
        decoded = self.decoder(target, memory, tgt_mask=causal)
        return self.output(self.output_norm(decoded))

    @torch.no_grad()
    def generate(self, image: Tensor, beam_size: int = 4, max_nodes: int = 64) -> list[list[str]]:
        """Return valid prefix programs, ranked by length-normalised log score."""
        if image.shape[0] != 1:
            raise ValueError("generate currently accepts one image at a time")
        device = image.device
        beams: list[tuple[list[int], GrammarState, float]] = [([self.vocabulary.bos_id], GrammarState(), 0.0)]
        completed: list[tuple[list[int], float]] = []
        for _ in range(self.max_length):
            candidates: list[tuple[list[int], GrammarState, float]] = []
            for ids, state, score in beams:
                if state.complete:
                    completed.append((ids, score))
                    continue
                logits = self(image, torch.tensor([ids], device=device))[0, -1]
                allowed = state.allowed_token_ids(self.vocabulary, max_nodes)
                masked = torch.full_like(logits, float("-inf"))
                masked[allowed] = logits[allowed]
                for token_id in torch.topk(F.log_softmax(masked, dim=0), min(beam_size, len(allowed))).indices.tolist():
                    next_state = GrammarState(**state.__dict__)
                    token = self.vocabulary.token(token_id)
                    next_state.consume(token, self.vocabulary)
                    candidates.append((ids + [token_id], next_state, score + float(F.log_softmax(masked, dim=0)[token_id])))
            candidates.sort(key=lambda item: item[2] / max(len(item[0]), 1), reverse=True)
            beams = candidates[:beam_size]
            if len(completed) >= beam_size:
                break
        completed.extend((ids, score) for ids, state, score in beams if state.complete)
        completed.sort(key=lambda item: item[1] / max(len(item[0]), 1), reverse=True)
        return [self.vocabulary.decode(ids[1:]) for ids, _ in completed[:beam_size]]


def sequence_cross_entropy(logits: Tensor, targets: Tensor, pad_id: int, length_weight: float = 0.02) -> Tensor:
    """Token loss plus a differentiable expected-length (MDL) regularizer."""
    token_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=pad_id)
    eos_probability = F.softmax(logits, dim=-1)[..., 2]
    # Probability the decoder has not stopped before each position.  Summing
    # this survival curve is the expected emitted length under its current
    # EOS policy, therefore minimising it genuinely prefers compact programs.
    survival = torch.cumprod(
        torch.cat((torch.ones_like(eos_probability[:, :1]), 1.0 - eos_probability[:, :-1]), dim=1),
        dim=1,
    )
    expected_length = survival.sum(dim=1).mean() / logits.shape[1]
    return token_loss + length_weight * expected_length
