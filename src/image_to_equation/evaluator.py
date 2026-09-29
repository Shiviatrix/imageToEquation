import re
import math
import torch
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Tuple, Union, Optional, OrderedDict as OrderedDictType

from image_to_equation.procedural_shaders import (
    MAX_FBM_OCTAVES,
    affine_gradient,
    bilinear_gradient,
    fractal_brownian_motion,
    gradient_noise2,
    over as shader_over,
    rgb as shader_rgb,
    shade as shader_shade,
    value_noise2,
    yeganeh_trig_noise,
    dexp_mask,
    polar_r,
    polar_theta,
)

sys.setrecursionlimit(10000)

# ==========================================
# 1. Lexer
# ==========================================

@dataclass
class Token:
    type: str
    value: str

class LexerError(Exception): pass
class ParserError(Exception): pass

class Lexer:
    TOKEN_REGEX = re.compile(
        r'(?P<NUM>\d+(\.\d+)?)|'
        r'(?P<ID>[a-zA-Z_]\w*)|'
        r'(?P<DOTDOT>\.\.)|'
        r'(?P<OP>[+\-*/^=,()])|'
        r'(?P<WS>\s+)'
    )

    def __init__(self, text: str):
        self.text = text
        self.tokens: List[Token] = []
        self.pos = 0
        self.tokenize()

    def tokenize(self):
        for match in self.TOKEN_REGEX.finditer(self.text):
            kind = match.lastgroup
            value = match.group()
            if kind == 'WS':
                continue
            self.tokens.append(Token(kind, value))
        self.tokens.append(Token('EOF', ''))

# ==========================================
# 2. AST Nodes
# ==========================================

class ASTNode: pass

@dataclass
class Constant(ASTNode):
    value: float

@dataclass
class Variable(ASTNode):
    name: str

@dataclass
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    op: str
    expr: ASTNode

@dataclass
class FuncCall(ASTNode):
    name: str
    arg: ASTNode

@dataclass
class MultiFuncCall(ASTNode):
    """A function call with two or more arguments.

    Unary calls keep their original AST shape so existing consumers that inspect
    ``FuncCall.arg`` remain compatible. Geometry and CSG helpers use this
    separate node rather than overloading comma expressions.
    """
    name: str
    args: List[ASTNode]

@dataclass
class SumLoop(ASTNode):
    var: str
    start: int
    end: int
    expr: ASTNode

@dataclass
class ProdLoop(ASTNode):
    var: str
    start: int
    end: int
    expr: ASTNode

@dataclass
class LetBinding(ASTNode):
    name: str
    value_expr: ASTNode
    body_expr: ASTNode

# ==========================================
# 3. Parser (Recursive Descent)
# ==========================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected_type: str = None, expected_value: str = None) -> Token:
        tok = self.current()
        if expected_type and tok.type != expected_type:
            raise ParserError(f"Expected type {expected_type}, got {tok.type} '{tok.value}'")
        if expected_value and tok.value != expected_value:
            raise ParserError(f"Expected value '{expected_value}', got '{tok.value}'")
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        ast = self.expr()
        if self.current().type != 'EOF':
            raise ParserError(f"Unexpected token at end: {self.current().value}")
        return ast

    def expr(self) -> ASTNode:
        node = self.term()
        while self.current().value in ('+', '-'):
            op = self.consume().value
            right = self.term()
            node = BinOp(op, node, right)
        return node

    def term(self) -> ASTNode:
        node = self.factor()
        while self.current().value in ('*', '/'):
            op = self.consume().value
            right = self.factor()
            node = BinOp(op, node, right)
        return node

    def factor(self) -> ASTNode:
        # Unary ops
        if self.current().value in ('+', '-'):
            op = self.consume().value
            return UnaryOp(op, self.factor())
        
        node = self.power()
        return node

    def power(self) -> ASTNode:
        node = self.primary()
        if self.current().value == '^':
            self.consume()
            right = self.factor()
            node = BinOp('^', node, right)
        return node

    def primary(self) -> ASTNode:
        tok = self.current()
        
        if tok.type == 'NUM':
            self.consume()
            return Constant(float(tok.value))
            
        elif tok.type == 'ID':
            name = tok.value
            self.consume()
            
            if self.current().value == '(':
                # Function call, sum, or prod
                self.consume(expected_value='(')
                
                if name in ('sum', 'prod', 'let'):
                    var_tok = self.consume(expected_type='ID')
                    if name == 'let':
                        self.consume(expected_value=',')
                        value_expr = self.expr()
                        self.consume(expected_value=',')
                        body_expr = self.expr()
                        self.consume(expected_value=')')
                        return LetBinding(var_tok.value, value_expr, body_expr)
                    else:
                        self.consume(expected_value='=')
                        start_tok = self.consume(expected_type='NUM')
                        self.consume(expected_type='DOTDOT')
                        end_tok = self.consume(expected_type='NUM')
                        self.consume(expected_value=',')
                        inner_expr = self.expr()
                        self.consume(expected_value=')')
                        
                        start_val = int(float(start_tok.value))
                        end_val = int(float(end_tok.value))
                        
                        if name == 'sum':
                            return SumLoop(var_tok.value, start_val, end_val, inner_expr)
                        else:
                            return ProdLoop(var_tok.value, start_val, end_val, inner_expr)
                else:
                    args = [self.expr()]
                    while self.current().value == ',':
                        self.consume(expected_value=',')
                        args.append(self.expr())
                    self.consume(expected_value=')')
                    if len(args) == 1:
                        return FuncCall(name, args[0])
                    return MultiFuncCall(name, args)
            else:
                if name == 'pi':
                    return Constant(math.pi)
                elif name == 'e':
                    return Constant(math.e)
                else:
                    return Variable(name)
                    
        elif tok.value == '(':
            self.consume()
            node = self.expr()
            self.consume(expected_value=')')
            return node
            
        else:
            raise ParserError(f"Unexpected token in primary: {tok.type} '{tok.value}'")

# ==========================================
# 4. Compiler (AST -> PyTorch Callable)
# ==========================================

class Compiler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Reusing scalar tensors is inexpensive but avoids thousands of tiny
        # allocations when a cached source contains repeated literals.  Keep it
        # bounded because fitting may explore many one-off numeric constants.
        self._constant_cache: OrderedDictType[float, torch.Tensor] = OrderedDict()
        self._constant_cache_size = 4096

    def _constant(self, value: float) -> torch.Tensor:
        value = float(value)
        cached = self._constant_cache.get(value)
        if cached is None:
            cached = torch.tensor(value, device=self.device, dtype=torch.float32)
            self._constant_cache[value] = cached
            if len(self._constant_cache) > self._constant_cache_size:
                self._constant_cache.popitem(last=False)
        else:
            self._constant_cache.move_to_end(value)
        return cached
        
    def compile(self, node: ASTNode) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
        if isinstance(node, Constant):
            value = self._constant(node.value)
            return lambda env: value
            
        elif isinstance(node, Variable):
            name = node.name
            return lambda env: env[name]
            
        elif isinstance(node, BinOp):
            left_fn = self.compile(node.left)
            right_fn = self.compile(node.right)
            
            if node.op == '+': return lambda env: left_fn(env) + right_fn(env)
            if node.op == '-': return lambda env: left_fn(env) - right_fn(env)
            if node.op == '*': return lambda env: left_fn(env) * right_fn(env)
            if node.op == '/': 
                return lambda env: left_fn(env) / right_fn(env)
            if node.op == '^':
                return lambda env: torch.pow(left_fn(env), right_fn(env))
                
        elif isinstance(node, UnaryOp):
            expr_fn = self.compile(node.expr)
            if node.op == '-': return lambda env: -expr_fn(env)
            if node.op == '+': return lambda env: expr_fn(env)
            
        elif isinstance(node, FuncCall):
            if (
                node.name == 'exp'
                and isinstance(node.arg, UnaryOp)
                and node.arg.op == '-'
                and isinstance(node.arg.expr, FuncCall)
                and node.arg.expr.name == 'exp'
            ):
                inner_fn = self.compile(node.arg.expr.arg)
                return lambda env: safe_double_exp(inner_fn(env))

            arg_fn = self.compile(node.arg)
            
            funcs = {
                'sin': torch.sin,
                'cos': torch.cos,
                'tan': torch.tan,
                'arcsin': lambda x: torch.asin(torch.clamp(x, -1.0, 1.0)),
                'asin': lambda x: torch.asin(torch.clamp(x, -1.0, 1.0)),
                'arccos': lambda x: torch.acos(torch.clamp(x, -1.0, 1.0)),
                'acos': lambda x: torch.acos(torch.clamp(x, -1.0, 1.0)),
                'arctan': torch.atan,
                'atan': torch.atan,
                'exp': torch.exp,
                'fmap': lambda x: 255.0 * torch.exp(-torch.exp(-1000.0 * torch.clamp(x, -1.0, 2.0))) * torch.pow(torch.abs(torch.clamp(x, -1.0, 2.0)) + 1e-9, torch.clamp(torch.exp(-1000.0 * (torch.clamp(x, -1.0, 2.0) - 1.0)), max=100.0)),
                'dexp': safe_double_exp,
                'log': lambda x: torch.log(torch.clamp(x, min=1e-12)),
                'abs': torch.abs,
                'sqrt': lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
                'sigmoid': torch.sigmoid,
                'tanh': torch.tanh,
                'softplus': torch.nn.functional.softplus,
            }
            if node.name not in funcs:
                raise ValueError(f"Unknown function: {node.name}")
            
            pt_func = funcs[node.name]
            return lambda env: pt_func(arg_fn(env))

        elif isinstance(node, MultiFuncCall):
            # The octave count is program structure rather than a learned
            # number. A literal bound keeps AST execution and MDL cost bounded.
            if node.name in ('fbm', 'fractal_brownian_motion'):
                if len(node.args) != 6:
                    raise ValueError(
                        f"Function '{node.name}' expects 6 arguments, got {len(node.args)}"
                    )
                octave_node = node.args[5]
                if not isinstance(octave_node, Constant) or not float(octave_node.value).is_integer():
                    raise ValueError(
                        "fbm's octave count must be an integer literal, not a fitted expression"
                    )
                octaves = int(octave_node.value)
                if not 1 <= octaves <= MAX_FBM_OCTAVES:
                    raise ValueError(
                        f"fbm's octave count must be in [1, {MAX_FBM_OCTAVES}]"
                    )
                x_fn, y_fn, frequency_fn, gain_fn, seed_fn = [
                    self.compile(arg) for arg in node.args[:5]
                ]

                def fbm(env):
                    return fractal_brownian_motion(
                        x_fn(env),
                        y_fn(env),
                        frequency_fn(env),
                        gain_fn(env),
                        seed_fn(env),
                        octaves,
                    )

                return fbm

            arg_fns = [self.compile(arg) for arg in node.args]

            def args(env):
                return [arg_fn(env) for arg_fn in arg_fns]

            def require_arity(expected: int):
                if len(arg_fns) != expected:
                    raise ValueError(
                        f"Function '{node.name}' expects {expected} arguments, got {len(arg_fns)}"
                    )

            if node.name in ('atan2', 'arctan2'):
                require_arity(2)
                return lambda env: torch.atan2(*args(env))
                
            if node.name in ('trignoise', 'yeganeh_trig_noise'):
                if len(node.args) != 5:
                    raise ValueError(
                        f"Function '{node.name}' expects 5 arguments, got {len(node.args)}"
                    )
                octave_node = node.args[4]
                if not isinstance(octave_node, Constant) or not float(octave_node.value).is_integer():
                    raise ValueError(
                        "trignoise's octave count must be an integer literal, not a fitted expression"
                    )
                octaves = int(octave_node.value)
                if not 1 <= octaves <= MAX_FBM_OCTAVES:
                    raise ValueError(
                        f"trignoise's octave count must be in [1, {MAX_FBM_OCTAVES}]"
                    )
                x_fn, y_fn, base_freq_fn, lacunarity_fn = [
                    self.compile(arg) for arg in node.args[:4]
                ]

                def trignoise_eval(env):
                    return yeganeh_trig_noise(
                        x_fn(env),
                        y_fn(env),
                        base_freq_fn(env),
                        lacunarity_fn(env),
                        octaves,
                    )
                return trignoise_eval
                
            if node.name in ('dexp_mask', 'dexpmask'):
                require_arity(2)
                return lambda env: dexp_mask(*args(env))
                
            if node.name == 'polar_r':
                require_arity(4)
                return lambda env: polar_r(*args(env))
                
            if node.name == 'polar_theta':
                require_arity(4)
                return lambda env: polar_theta(*args(env))

            if node.name in ('affine', 'agrad', 'linear_gradient'):
                require_arity(5)
                return lambda env: affine_gradient(*args(env))

            if node.name in ('bilinear', 'bilinear_gradient'):
                require_arity(6)
                return lambda env: bilinear_gradient(*args(env))

            if node.name in ('valuenoise', 'vnoise', 'value_noise'):
                require_arity(4)
                return lambda env: value_noise2(*args(env))

            if node.name in ('gradnoise', 'gnoise', 'gradient_noise'):
                require_arity(4)
                return lambda env: gradient_noise2(*args(env))

            if node.name == 'rgb':
                require_arity(3)
                return lambda env: shader_rgb(*args(env))

            if node.name in ('shade', 'paint'):
                require_arity(4)
                return lambda env: shader_shade(*args(env))

            if node.name in ('over', 'composite'):
                require_arity(3)
                return lambda env: shader_over(*args(env))

            if node.name in ('sdfcolour', 'sdf_color'):
                if len(arg_fns) < 5 or (len(arg_fns) - 3) % 2:
                    raise ValueError(
                        "sdfcolour expects base, sharpness, edge, then one or more (distance, rgb) pairs"
                    )

                def sdfcolour(env):
                    values = args(env)
                    background, sharpness, edge = values[:3]
                    distances = torch.broadcast_tensors(*values[3::2])
                    colours = values[4::2]
                    field_stack = torch.stack(distances, dim=0)
                    sharpness = torch.clamp(torch.abs(sharpness), min=1e-4)
                    weights = torch.softmax(-sharpness * field_stack, dim=0)
                    union_distance = -torch.logsumexp(-sharpness * field_stack, dim=0) / sharpness
                    height, width = field_stack.shape[-2:]
                    if background.ndim == 1 and background.shape[0] == 3:
                        background = background.view(3, 1, 1)
                    base_rgb = torch.broadcast_to(background, (3, height, width))

                    def as_rgb(colour):
                        if colour.ndim == 1 and colour.shape[0] == 3:
                            colour = colour.view(3, 1, 1)
                        return torch.broadcast_to(colour, base_rgb.shape)

                    colour_stack = torch.stack(
                        [as_rgb(colour) for colour in colours], dim=0
                    )
                    layer_colour = (weights * colour_stack).sum(dim=0)
                    alpha = torch.sigmoid(
                        -union_distance / torch.clamp(torch.abs(edge), min=1e-5)
                    )
                    return layer_colour * alpha + base_rgb * (1.0 - alpha)

                return sdfcolour

            if node.name == 'hypot':
                require_arity(2)
                return lambda env: torch.hypot(*args(env))

            if node.name in ('smin', 'smooth_union'):
                require_arity(3)

                def smin(env):
                    a, b, sharpness = args(env)
                    a, b = torch.broadcast_tensors(a, b)
                    sharpness = torch.clamp(sharpness, min=1e-4)
                    return -torch.logsumexp(torch.stack((-sharpness * a, -sharpness * b)), dim=0) / sharpness

                return smin

            if node.name in ('smax', 'smooth_intersection'):
                require_arity(3)

                def smax(env):
                    a, b, sharpness = args(env)
                    a, b = torch.broadcast_tensors(a, b)
                    sharpness = torch.clamp(sharpness, min=1e-4)
                    return torch.logsumexp(torch.stack((sharpness * a, sharpness * b)), dim=0) / sharpness

                return smax

            if node.name in ('sdfmask', 'softmask'):
                require_arity(2)

                def sdfmask(env):
                    distance, softness = args(env)
                    return torch.sigmoid(-distance / torch.clamp(torch.abs(softness), min=1e-5))

                return sdfmask

            if node.name == 'smoothstep':
                require_arity(3)

                def smoothstep(env):
                    low, high, value = args(env)
                    width = torch.clamp(high - low, min=1e-5)
                    t = torch.clamp((value - low) / width, 0.0, 1.0)
                    return t * t * (3.0 - 2.0 * t)

                return smoothstep

            if node.name == 'mix':
                require_arity(3)

                def mix(env):
                    a, b, amount = args(env)
                    return a * (1.0 - amount) + b * amount

                return mix

            if node.name == 'clamp':
                require_arity(3)

                def clamp(env):
                    value, low, high = args(env)
                    return torch.clamp(value, min=low, max=high)

                return clamp

            if node.name == 'disk':
                require_arity(5)

                def disk_sdf(env):
                    x, y, cx, cy, radius = args(env)
                    return torch.hypot(x - cx, y - cy) - radius

                return disk_sdf

            if node.name in ('box', 'rect'):
                require_arity(6)

                def box_sdf(env):
                    x, y, cx, cy, half_width, half_height = args(env)
                    qx = torch.abs(x - cx) - half_width
                    qy = torch.abs(y - cy) - half_height
                    outside = torch.hypot(torch.clamp(qx, min=0.0), torch.clamp(qy, min=0.0))
                    inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
                    return outside + inside

                return box_sdf

            if node.name == 'superellipse':
                require_arity(8)

                def superellipse_sdf(env):
                    x, y, cx, cy, rx, ry, power, theta = args(env)
                    cosine, sine = torch.cos(theta), torch.sin(theta)
                    dx, dy = x - cx, y - cy
                    qx = cosine * dx + sine * dy
                    qy = -sine * dx + cosine * dy
                    rx = torch.clamp(torch.abs(rx), min=1e-5)
                    ry = torch.clamp(torch.abs(ry), min=1e-5)
                    power = torch.clamp(power, min=1.05, max=8.0)
                    scale = torch.minimum(rx, ry)
                    normalized = (torch.abs(qx) / rx).pow(power) + (torch.abs(qy) / ry).pow(power)
                    return normalized.pow(1.0 / power) * scale - scale

                return superellipse_sdf

            if node.name == 'capsule':
                require_arity(7)

                def capsule_sdf(env):
                    x, y, cx, cy, half_length, radius, theta = args(env)
                    cosine, sine = torch.cos(theta), torch.sin(theta)
                    dx, dy = x - cx, y - cy
                    qx = cosine * dx + sine * dy
                    qy = -sine * dx + cosine * dy
                    outside_x = torch.clamp(torch.abs(qx) - torch.clamp(half_length, min=0.0), min=0.0)
                    return torch.hypot(outside_x, qy) - torch.clamp(torch.abs(radius), min=1e-5)

                return capsule_sdf

            if node.name == 'wedge':
                require_arity(7)

                def wedge_sdf(env):
                    x, y, cx, cy, length, width, theta = args(env)
                    cosine, sine = torch.cos(theta), torch.sin(theta)
                    dx, dy = x - cx, y - cy
                    qx = cosine * dx + sine * dy
                    qy = -sine * dx + cosine * dy
                    length = torch.clamp(torch.abs(length), min=1e-5)
                    width = torch.clamp(torch.abs(width), min=1e-5)
                    slope = width / (2.0 * length)
                    diagonal_scale = torch.sqrt(1.0 + slope.square())
                    fields = torch.stack((
                        -qx - length / 2.0,
                        qx - length / 2.0,
                        (qy - width / 4.0 + slope * qx) / diagonal_scale,
                        (-qy - width / 4.0 + slope * qx) / diagonal_scale,
                    ))
                    return torch.max(fields, dim=0).values

                return wedge_sdf

            if node.name in ('rounded_box', 'roundbox'):
                require_arity(8)

                def rounded_box_sdf(env):
                    x, y, cx, cy, hx, hy, radius, theta = args(env)
                    cosine, sine = torch.cos(theta), torch.sin(theta)
                    dx, dy = x - cx, y - cy
                    qx = cosine * dx + sine * dy
                    qy = -sine * dx + cosine * dy
                    hx = torch.clamp(torch.abs(hx), min=1e-5)
                    hy = torch.clamp(torch.abs(hy), min=1e-5)
                    radius = torch.clamp(radius, min=0.0)
                    bx, by = torch.abs(qx) - hx, torch.abs(qy) - hy
                    outside = torch.hypot(torch.clamp(bx, min=0.0), torch.clamp(by, min=0.0))
                    inside = torch.minimum(torch.maximum(bx, by), torch.zeros_like(bx))
                    return outside + inside - radius

                return rounded_box_sdf

            raise ValueError(f"Unknown function: {node.name}")
            
        elif isinstance(node, SumLoop):
            if node.end - node.start > 500:
                raise ValueError("Loop bound too large (>500)")
            inner_fn = self.compile(node.expr)
            start, end, var_name = node.start, node.end, node.var
            # Loop bounds are syntax-level integers, not trainable values.  Keep
            # their device scalars with the compiled expression so repeated GD
            # renders do not allocate one scalar per octave per iteration.
            loop_values = tuple(self._constant(float(i)) for i in range(start, end + 1))
            zero = self._constant(0.0)
            
            def sum_eval(env):
                # Preserve sequential accumulation order.  A stack/reduce is
                # faster for tiny loops but can multiply H*W peak memory and
                # changes floating-point reduction order for long fBm sums.
                old_val = env.get(var_name)
                total = zero
                try:
                    for loop_value in loop_values:
                        env[var_name] = loop_value
                        total = total + inner_fn(env)
                finally:
                    if old_val is not None:
                        env[var_name] = old_val
                    else:
                        env.pop(var_name, None)
                return total
            return sum_eval
            
        elif isinstance(node, ProdLoop):
            if node.end - node.start > 500:
                raise ValueError("Loop bound too large (>500)")
            inner_fn = self.compile(node.expr)
            start, end, var_name = node.start, node.end, node.var
            loop_values = tuple(self._constant(float(i)) for i in range(start, end + 1))
            one = self._constant(1.0)
            
            def prod_eval(env):
                old_val = env.get(var_name)
                total = one
                try:
                    for loop_value in loop_values:
                        env[var_name] = loop_value
                        total = total * inner_fn(env)
                finally:
                    if old_val is not None:
                        env[var_name] = old_val
                    else:
                        env.pop(var_name, None)
                return total
            return prod_eval

        elif isinstance(node, LetBinding):
            value_fn = self.compile(node.value_expr)
            body_fn = self.compile(node.body_expr)
            var_name = node.name
            
            def let_eval(env):
                old_val = env.get(var_name)
                env[var_name] = value_fn(env)
                try:
                    return body_fn(env)
                finally:
                    if old_val is not None:
                        env[var_name] = old_val
                    else:
                        env.pop(var_name, None)
            return let_eval

        raise NotImplementedError(f"Unsupported AST node: {type(node)}")

# ==========================================
# 5. Evaluators
# ==========================================

def safe_exp(z: torch.Tensor) -> torch.Tensor:
    """Float32-safe exponential with an exactly-zero tail beyond visual range."""
    return torch.exp(torch.clamp(z, max=80.0))


def safe_double_exp(z: torch.Tensor) -> torch.Tensor:
    """Computes ``exp(-exp(z))`` without overflowing the inner exponential."""
    ez = safe_exp(z)
    return torch.exp(-ez)

def yeganeh_clamp(x: torch.Tensor) -> torch.Tensor:
    """
    F(x) = floor(255 * exp(-exp(-1000*x)) * abs(x) * exp(-exp(1000*(x-1))))
    """
    term1 = safe_double_exp(-1000.0 * x)
    term2 = safe_double_exp(1000.0 * (x - 1.0))
    val = 255.0 * term1 * torch.abs(x) * term2
    return torch.floor(val).clamp(0, 255).to(torch.uint8)


class _TorchCompileFallback:
    """Call an optional ``torch.compile`` graph, falling back to eager safely.

    ``torch.compile`` is intentionally opt-in because its first invocation can
    cost far more than a small render.  Some valid ASTs also contain dynamic
    dictionary access or Python loops that a particular PyTorch backend may not
    lower.  A failure must not make equation evaluation unavailable, so the
    wrapper permanently returns to the eager callable after the first error.
    """

    def __init__(self, eager_fn: Callable, compiled_fn: Callable):
        self._eager_fn = eager_fn
        self._compiled_fn: Optional[Callable] = compiled_fn
        self.fallback_reason: Optional[str] = None

    def __call__(self, env: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._compiled_fn is None:
            return self._eager_fn(env)
        try:
            return self._compiled_fn(env)
        except Exception as exc:  # pragma: no cover - backend dependent
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self._compiled_fn = None
            return self._eager_fn(env)

class GridEvaluator:
    def __init__(
        self,
        expression_cache_size: int = 128,
        enable_torch_compile: bool = False,
        torch_compile_mode: Optional[str] = "reduce-overhead",
        torch_compile_backend: Optional[str] = None,
    ):
        """Create an evaluator with bounded source and coordinate caches.

        ``enable_torch_compile`` is opt-in and leaves the eager evaluator as
        the default.  It can fuse a stable, repeatedly-rendered expression, but
        startup compilation is usually a loss for one-off equations.  If a
        backend cannot lower an otherwise valid AST, that source transparently
        continues in eager mode.
        """
        if expression_cache_size < 0:
            raise ValueError("expression_cache_size must be non-negative")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compiler = Compiler()
        self._grid_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, ...]] = {}
        self.expression_cache_size = expression_cache_size
        self.enable_torch_compile = enable_torch_compile
        self.torch_compile_mode = torch_compile_mode
        self.torch_compile_backend = torch_compile_backend
        self._expression_cache: OrderedDictType[str, Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = OrderedDict()
        self._expression_cache_hits = 0
        self._expression_cache_misses = 0

    def clear_expression_cache(self) -> None:
        """Forget parsed/compiled source expressions while retaining grids."""
        self._expression_cache.clear()
        self._expression_cache_hits = 0
        self._expression_cache_misses = 0

    def clear_grid_cache(self) -> None:
        """Release coordinate-grid tensors retained by this evaluator."""
        self._grid_cache.clear()

    def cache_info(self) -> Dict[str, int]:
        """Return inexpensive cache instrumentation for benchmarks and tests."""
        return {
            "expression_entries": len(self._expression_cache),
            "expression_capacity": self.expression_cache_size,
            "expression_hits": self._expression_cache_hits,
            "expression_misses": self._expression_cache_misses,
            "grid_entries": len(self._grid_cache),
        }

    def _optional_torch_compile(self, eager_fn: Callable) -> Callable:
        if not self.enable_torch_compile or not hasattr(torch, "compile"):
            return eager_fn
        try:
            kwargs: Dict[str, Any] = {"fullgraph": False, "dynamic": False}
            if self.torch_compile_mode is not None:
                kwargs["mode"] = self.torch_compile_mode
            if self.torch_compile_backend is not None:
                kwargs["backend"] = self.torch_compile_backend
            return _TorchCompileFallback(eager_fn, torch.compile(eager_fn, **kwargs))
        except Exception:
            # Compilation configuration errors are non-fatal.  The original
            # callable has exactly the historical evaluation semantics.
            return eager_fn

    def _compiled_expression(self, eq: str) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
        cached = self._expression_cache.get(eq)
        if cached is not None:
            self._expression_cache_hits += 1
            self._expression_cache.move_to_end(eq)
            return cached

        self._expression_cache_misses += 1
        ast = Parser(Lexer(eq).tokens).parse()
        compiled_fn = self._optional_torch_compile(self.compiler.compile(ast))

        if self.expression_cache_size:
            self._expression_cache[eq] = compiled_fn
            self._expression_cache.move_to_end(eq)
            while len(self._expression_cache) > self.expression_cache_size:
                self._expression_cache.popitem(last=False)
        return compiled_fn

    def _grid(self, width: int, height: int) -> Tuple[torch.Tensor, ...]:
        key = (width, height)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached

        x_lin = torch.linspace(0, 1, width, device=self.device)
        y_lin = torch.linspace(0, 1, height, device=self.device)
        y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing='ij')
        grid = (
            x_grid.unsqueeze(0),
            y_grid.unsqueeze(0),
            torch.tensor([0.0, 1.0, 2.0], device=self.device).view(3, 1, 1),
            torch.tensor(float(width), device=self.device),
            torch.tensor(float(height), device=self.device),
        )
        self._grid_cache[key] = grid
        return grid

    def evaluate(
        self,
        eq: str,
        width: int,
        height: int,
        differentiable: bool = False,
        params: Optional[Dict[str, torch.Tensor]] = None,
        render_mode: str = 'legacy',
        sdf_softness: float = 0.01,
    ) -> torch.Tensor:
        if render_mode not in {'legacy', 'sdf', 'linear'}:
            raise ValueError("render_mode must be 'legacy', 'sdf', or 'linear'")
        if sdf_softness <= 0:
            raise ValueError('sdf_softness must be positive')
        compiled_fn = self._compiled_expression(eq)
        
        X, Y, V, W, H = self._grid(width, height)
        
        env = {
            'x': X,
            'y': Y,
            'v': V,
            'W': W,
            'H': H,
        }
        
        if params is not None:
            for k, v in params.items():
                env[k] = v.to(self.device)
        
        raw_result = compiled_fn(env)
        # ``rgb(1, 0, 0)`` is a valid spatially constant shader.  Keep its
        # channel axis separate before broadcasting over the requested canvas.
        if raw_result.ndim == 1 and raw_result.shape[0] == 3:
            raw_result = raw_result.view(3, 1, 1)
        raw_result = torch.broadcast_to(raw_result, (3, height, width))
        
        if render_mode == 'legacy' and differentiable:
            term1 = safe_double_exp(-1000.0 * raw_result)
            term2 = safe_double_exp(1000.0 * (raw_result - 1.0))
            val = 255.0 * term1 * torch.abs(raw_result) * term2
            img_rgb = val.clamp(0, 255)
        elif render_mode == 'legacy':
            img_rgb = yeganeh_clamp(raw_result)
        else:
            if render_mode == 'sdf':
                normalized = torch.sigmoid(-raw_result / sdf_softness)
            else:
                normalized = torch.clamp(raw_result, 0.0, 1.0)
            img_rgb = 255.0 * normalized
            if not differentiable:
                img_rgb = torch.round(img_rgb).to(torch.uint8)
            
        img_rgb = img_rgb.permute(1, 2, 0)
        
        return img_rgb

class BatchEvaluator:
    def __init__(self):
        self.grid_evaluator = GridEvaluator()
        self.device = self.grid_evaluator.device

    def evaluate_batch(self, eqs: List[str], width: int, height: int) -> torch.Tensor:
        results = []
        for eq in eqs:
            img = self.grid_evaluator.evaluate(eq, width, height)
            results.append(img)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        return torch.stack(results)
