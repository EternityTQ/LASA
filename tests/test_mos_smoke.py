"""
Smoke tests for MOS-Attack stability fixes (mos.py).

All tests use a pure-Python mock of torch so no GPU/CPU install is needed.
"""

import sys, os, math, types

# ═══════════════════════════════════════════════════════════════════════════
# Minimal but complete-enough torch mock
# ═══════════════════════════════════════════════════════════════════════════

class DT:
    """Fake dtype."""
    def __init__(self, name): self._name = name
    def __repr__(self): return f"torch.{self._name}"

_int32  = DT('int32')
_float32 = DT('float32')

class Tensor:
    """Tensor mock supporting both 1-D and logical 2-D shapes."""
    def __init__(self, data, device='cpu', dtype=_float32, requires_grad=False, shape=None):
        if isinstance(data, Tensor):
            self._v = list(data._v)
        elif isinstance(data, (list, tuple)):
            self._v = [int(x) if dtype is _int32 else float(x) for x in data]
        else:
            self._v = [int(data) if dtype is _int32 else float(data)]
        self.device = device
        self.dtype = dtype
        self.shape = shape if shape is not None else (len(self._v),)

    # -- helpers -----------------------------------------------------------
    def numel(self):          return len(self._v)
    def reshape(self, *s):    t = Tensor(self._v, device=self.device, dtype=self.dtype); t.shape = s; return t
    def flatten(self):        return Tensor(self._v, device=self.device, dtype=self.dtype)
    def unsqueeze(self, d):   t = Tensor(self._v, device=self.device, dtype=self.dtype); t.shape = (1, len(self._v)); return t
    def squeeze(self, d=None):return Tensor(self._v, device=self.device, dtype=self.dtype)
    def clone(self):          return Tensor(self._v, device=self.device, dtype=self.dtype)
    def detach(self):         return Tensor(self._v, device=self.device, dtype=self.dtype)
    def to(self, dev):        return Tensor(self._v, device=dev, dtype=self.dtype)
    def float(self):          return Tensor(self._v, device=self.device, dtype=_float32)
    def long(self):           return Tensor([int(x) for x in self._v], device=self.device, dtype=_int32)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            r, c = idx
            nrows = self.shape[0] if len(self.shape) >= 2 else 1
            ncols = self.shape[1] if len(self.shape) >= 2 else len(self._v)
            # (int, int) → scalar
            if isinstance(r, int) and isinstance(c, int):
                return self._v[r * ncols + c]
            # (slice, int) → column slice → 1D tensor
            if isinstance(r, slice) and isinstance(c, int):
                rows = list(range(*r.indices(nrows)))
                return Tensor([self._v[rr * ncols + c] for rr in rows], device=self.device, dtype=self.dtype)
            # (int, slice) → row slice → 1D tensor
            if isinstance(r, int) and isinstance(c, slice):
                cols = list(range(*c.indices(ncols)))
                return Tensor([self._v[r * ncols + cc] for cc in cols], device=self.device, dtype=self.dtype)
            # (int, list) → fancy column indexing on one row → 1D tensor
            if isinstance(r, int) and isinstance(c, list):
                return Tensor([self._v[r * ncols + cc] for cc in c], device=self.device, dtype=self.dtype)
            # (list, int) → fancy row indexing on one column → 1D tensor
            if isinstance(r, list) and isinstance(c, int):
                return Tensor([self._v[rr * ncols + c] for rr in r], device=self.device, dtype=self.dtype)
            # (list, slice) → fancy row indexing, all columns → 2D tensor
            if isinstance(r, list) and isinstance(c, slice):
                cols = list(range(*c.indices(ncols)))
                vals = []
                for rr in r:
                    vals.extend([self._v[rr * ncols + cc] for cc in cols])
                return Tensor(vals, device=self.device, dtype=self.dtype, shape=(len(r), len(cols)))
            # (slice, list) → all rows, fancy columns → 2D tensor
            if isinstance(r, slice) and isinstance(c, list):
                rows = list(range(*r.indices(nrows)))
                vals = []
                for rr in rows:
                    vals.extend([self._v[rr * ncols + cc] for cc in c])
                return Tensor(vals, device=self.device, dtype=self.dtype, shape=(len(rows), len(c)))
            # fallback
            return Tensor([self._v[0]], device=self.device, dtype=self.dtype)
        # single integer index on a 2D-like tensor → return 1D row slice
        if isinstance(idx, int) and len(self.shape) >= 2:
            ncols = self.shape[1]
            start = idx * ncols
            return Tensor(self._v[start:start + ncols], device=self.device, dtype=self.dtype)
        if isinstance(idx, int):
            # Return 0-d Tensor preserving dtype
            return Tensor([self._v[idx]], device=self.device, dtype=self.dtype)
        if isinstance(idx, list):
            # Fancy indexing: list of row indices on a 2D tensor → (len(idx), D)
            if len(self.shape) >= 2:
                ncols = self.shape[1]
                vals = []
                for i in idx:
                    vals.extend(self._v[i * ncols:(i + 1) * ncols])
                return Tensor(vals, device=self.device, dtype=self.dtype,
                             shape=(len(idx), ncols))
            return Tensor([self._v[i] for i in idx], device=self.device, dtype=self.dtype)
        if isinstance(idx, Tensor):
            if idx.dtype == _int32 or idx.dtype == DT('bool'):
                return self.__getitem__([int(i) for i in idx._v])
        if isinstance(idx, slice):
            return Tensor(self._v[idx], device=self.device, dtype=self.dtype)
        return self._v[idx]

    def __setitem__(self, idx, val):
        if isinstance(val, Tensor): self._v[idx] = val.item() if val.numel() == 1 else val._v[0]
        else: self._v[idx] = float(val)

    def __len__(self): return len(self._v)
    def __iter__(self):
        for i in range(len(self._v)):
            yield self[i]  # returns 0-D Tensor via __getitem__

    # arith
    def __add__(self, o): return _binop(self, o, lambda a, b: a + b)
    def __radd__(self, o):return _binop(self, o, lambda a, b: b + a)
    def __sub__(self, o): return _binop(self, o, lambda a, b: a - b)
    def __rsub__(self, o):return _binop(self, o, lambda a, b: b - a)
    def __mul__(self, o): return _binop(self, o, lambda a, b: a * b)
    def __rmul__(self, o):return _binop(self, o, lambda a, b: b * a)
    def __truediv__(self, o):return _binop(self, o, lambda a, b: a / (b + 1e-30))
    def __rtruediv__(self, o):return _binop(self, o, lambda a, b: b / (a + 1e-30))
    def __neg__(self):     return Tensor([-x for x in self._v], device=self.device, dtype=self.dtype)
    def __pos__(self):     return self
    def __pow__(self, o):
        if isinstance(o, (int, float)): return Tensor([x ** o for x in self._v], device=self.device, dtype=self.dtype)
        return NotImplemented
    def square(self):      return Tensor([x * x for x in self._v], device=self.device, dtype=self.dtype)
    def sqrt(self):        return Tensor([math.sqrt(max(x, 0)) for x in self._v], device=self.device, dtype=self.dtype)
    def abs(self):         return Tensor([abs(x) for x in self._v], device=self.device, dtype=self.dtype)
    def sign(self):        return Tensor([1.0 if x >= 0 else -1.0 for x in self._v], device=self.device, dtype=self.dtype)

    def relu(self):
        class _R:
            def __init__(self, v, dev, dt):
                self._v = [max(x, 0) for x in v]; self.device = dev; self.dtype = dt
        return _R(self._v, self.device, self.dtype)

    def __eq__(self, o): return _binop(self, o, lambda a, b: a == b)
    def __ne__(self, o): return _binop(self, o, lambda a, b: a != b)
    def __lt__(self, o): return _binop(self, o, lambda a, b: a < b)
    def __le__(self, o): return _binop(self, o, lambda a, b: a <= b)
    def __gt__(self, o): return _binop(self, o, lambda a, b: a > b)
    def __ge__(self, o): return _binop(self, o, lambda a, b: a >= b)

    def item(self):
        if len(self._v) == 1:
            if self.dtype is _int32:
                return int(self._v[0])
            return float(self._v[0])
        raise ValueError("only one element tensors can be converted to Python scalars")

    def __bool__(self):
        if len(self._v) != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self._v[0])

    def copy_(self, src):
        self._v = list(src._v) if isinstance(src, Tensor) else [float(src)]

    def new_tensor(self, data):
        if isinstance(data, (int, float)): data = [data]
        return Tensor(data, device=self.device)

    def unsqueeze_(self, d):
        self.shape = (1, len(self._v)); return self

    # reductions that return Tensor
    def sum(self, dim=None):
        if dim is None: return Tensor([sum(self._v)])
        if dim == 0 and len(self.shape) >= 2:
            nrows, ncols = self.shape[0], self.shape[1]
            sums = [sum(self._v[r * ncols + c] for r in range(nrows)) for c in range(ncols)]
            return Tensor(sums)
        return Tensor([sum(self._v)])

    def mean(self, dim=None):
        if dim is None: return Tensor([sum(self._v) / max(len(self._v), 1)])
        if dim == 0 and len(self.shape) >= 2:
            nrows, ncols = self.shape[0], self.shape[1]
            means = [sum(self._v[r * ncols + c] for r in range(nrows)) / nrows for c in range(ncols)]
            return Tensor(means)
        return Tensor([sum(self._v) / max(len(self._v), 1)])

    def max(self, dim=None):
        if dim is None: return Tensor([max(self._v)])
        return (Tensor([max(self._v)]), Tensor([self._v.index(max(self._v))], dtype=_int32))

    def min(self, dim=None):
        if dim is None: return Tensor([min(self._v)])
        return (Tensor([min(self._v)]), Tensor([self._v.index(min(self._v))], dtype=_int32))

    def std(self, dim=None, correction=0, keepdim=False):
        n = max(len(self._v) - correction, 1)
        m = sum(self._v) / len(self._v)
        var = sum((x - m) ** 2 for x in self._v) / n
        s = math.sqrt(var)
        return Tensor([s])

    def nonzero(self):
        idx = [i for i, x in enumerate(self._v) if x != 0]
        t = Tensor(idx, dtype=_int32)
        t.shape = (len(idx), 1)
        return t

    def argsort(self, descending=False):
        s = sorted(range(len(self._v)), key=lambda i: self._v[i], reverse=descending)
        return Tensor(s, dtype=_int32)

    def all(self):   return all(self._v)
    def any(self):   return any(self._v)

    def scatter_(self, dim, index, value):
        return self

    def clamp(self, min_val=None, max_val=None):
        out = []
        for x in self._v:
            v = x
            if min_val is not None: v = max(v, min_val)
            if max_val is not None: v = min(v, max_val)
            out.append(v)
        return Tensor(out)


def _binop(a, b, op):
    if isinstance(b, Tensor):
        # Scalar broadcast: (N,) op (1,) → (N,)
        if len(b._v) == 1 and len(a._v) > 1:
            return Tensor([op(av, b._v[0]) for av in a._v], device=a.device, dtype=a.dtype, shape=a.shape)
        # Scalar broadcast: (1,) op (N,) → (N,)
        if len(a._v) == 1 and len(b._v) > 1:
            return Tensor([op(a._v[0], bv) for bv in b._v], device=a.device, dtype=a.dtype, shape=b.shape)
        # Row-wise broadcast: (R, C) op (C,) → (R, C)
        if len(a.shape) >= 2 and len(b.shape) == 1 and a.shape[1] == b.shape[0]:
            ncols = a.shape[1]
            nrows = a.shape[0]
            out = []
            for r in range(nrows):
                base = r * ncols
                for c in range(ncols):
                    out.append(op(a._v[base + c], b._v[c]))
            return Tensor(out, device=a.device, dtype=a.dtype, shape=a.shape)
        # (C,) op (R, C) → (R, C)
        if len(b.shape) >= 2 and len(a.shape) == 1 and b.shape[1] == a.shape[0]:
            return _binop(b, a, lambda x, y: op(y, x))
        return Tensor([op(av, bv) for av, bv in zip(a._v, b._v)], device=a.device, dtype=a.dtype, shape=a.shape)
    if isinstance(b, (int, float)):
        return Tensor([op(av, b) for av in a._v], device=a.device, dtype=a.dtype, shape=a.shape)
    return NotImplemented

# Torch module-level functions
def _tensor(data, **kw):
    if isinstance(data, (int, float)): return Tensor([data], **kw)
    if isinstance(data, list):
        dtype = kw.get('dtype', _float32)
        return Tensor(data, device=kw.get('device', 'cpu'), dtype=dtype)
    if isinstance(data, Tensor): return data
    return Tensor([0.0], **kw)

def _tensor_isfinite(t):
    if isinstance(t, Tensor):
        return Tensor([int(math.isfinite(x)) for x in t._v])
    return all(math.isfinite(x) for x in (t if hasattr(t, '__iter__') else [t]))

def _norm(t, p=2, dim=None, keepdim=False):
    if isinstance(t, Tensor):
        if dim is None:
            return Tensor([math.sqrt(sum(x * x for x in t._v))])
        if dim == 1 and len(t.shape) >= 2:
            # per-row L2 norm
            nrows, ncols = t.shape[0], t.shape[1]
            norms = []
            for r in range(nrows):
                base = r * ncols
                norms.append(math.sqrt(sum(t._v[base + c] ** 2 for c in range(ncols))))
            result = Tensor(norms)
            if keepdim:
                result.shape = (nrows, 1)
            return result
        if dim == 1:
            return Tensor([math.sqrt(sum(x * x for x in t._v))])
    return Tensor([0.0])

def _dot(a, b):
    """dot product → 0-d Tensor"""
    return Tensor([sum(av * bv for av, bv in zip(a._v, b._v))])

def _torch_mean(t, dim=None):
    if isinstance(t, Tensor):
        if dim == 0 and len(t.shape) >= 2:
            # mean across rows → (C,) tensor
            nrows, ncols = t.shape[0], t.shape[1]
            means = []
            for c in range(ncols):
                col_sum = sum(t._v[r * ncols + c] for r in range(nrows))
                means.append(col_sum / nrows)
            return Tensor(means)
        if dim is None:
            return Tensor([sum(t._v) / max(len(t._v), 1)])
        return Tensor([sum(t._v) / max(len(t._v), 1)])
    return 0.0

def _torch_std(t, dim=None, correction=0, keepdim=False):
    if isinstance(t, Tensor):
        if dim == 0 and len(t.shape) >= 2:
            nrows, ncols = t.shape[0], t.shape[1]
            n = max(nrows - correction, 1)
            stds = []
            for c in range(ncols):
                col_vals = [t._v[r * ncols + c] for r in range(nrows)]
                m = sum(col_vals) / nrows
                var = sum((x - m) ** 2 for x in col_vals) / n
                stds.append(math.sqrt(var))
            return Tensor(stds)
        if dim is None:
            n = max(len(t._v) - correction, 1)
            m = sum(t._v) / len(t._v)
            return Tensor([math.sqrt(sum((x - m) ** 2 for x in t._v) / n)])
        n = max(len(t._v) - correction, 1)
        m = sum(t._v) / len(t._v)
        return Tensor([math.sqrt(sum((x - m) ** 2 for x in t._v) / n)])
    return Tensor([0.0])

def _all(t):
    """torch.all() equivalent"""
    if isinstance(t, Tensor):
        return all(t._v)
    return bool(t)

def _any(t):
    """torch.any() equivalent"""
    if isinstance(t, Tensor):
        return any(t._v)
    return bool(t)

def _quantile(t, q, dim=None):
    if isinstance(t, Tensor):
        sv = sorted(t._v)
        idx = min(int(q * (len(sv) - 1)), len(sv) - 1) if sv else 0
        return Tensor([sv[idx]])
    return Tensor([0.0])

def _clamp(t, min=None, max=None):
    out = []
    for x in t._v:
        v = x
        if min is not None: v = max(v, min)
        if max is not None: v = min(v, max)
        out.append(v)
    return Tensor(out)

def _relu(t):
    if isinstance(t, Tensor):
        return Tensor([max(x, 0) for x in t._v])
    # Handle custom relu wrapper from Tensor.relu()
    return Tensor([max(x, 0) for x in t._v])

def _argmax(t, dim=None):
    if isinstance(t, Tensor) and t._v:
        m = max(t._v)
        return t._v.index(m)
    return 0

def _sign(t):
    if isinstance(t, Tensor):
        return Tensor([1.0 if x >= 0 else -1.0 for x in t._v])
    return Tensor([1.0])

def _cat(tensors, dim=0):
    all_v = []
    dev = 'cpu'
    for t in tensors:
        all_v.extend(t._v)
        dev = t.device
    return Tensor(all_v, device=dev)

def _stack(tensors, dim=0):
    return Tensor(tensors[0]._v)  # simplified

def _gather(t, dim, index):
    return t

def _empty(*args, **kw):
    if args and isinstance(args[0], tuple):
        n = args[0][0] if args[0] else 0
    elif args:
        n = args[0] if isinstance(args[0], int) else 0
    else:
        n = 0
    return Tensor([0.0] * n, device=kw.get('device', 'cpu'), dtype=kw.get('dtype', _float32))

def _zeros(*args, **kw):
    if args and isinstance(args[0], int):
        n = args[0]
    elif args and isinstance(args[0], tuple):
        n = args[0][0] if args[0] else 1
    else:
        n = 1
    return Tensor([0.0] * n, device=kw.get('device', 'cpu'), dtype=kw.get('dtype', _float32))

def _ones(*args, **kw):
    n = args[0] if (args and isinstance(args[0], int)) else 1
    return Tensor([1.0] * n, device=kw.get('device', 'cpu'))

def _zeros_like(t, **kw):
    return Tensor([0.0] * len(t._v), device=t.device, dtype=t.dtype)

def _randn_like(t, **kw):
    import random as _r
    return Tensor([_r.gauss(0, 1) for _ in t._v], device=t.device, dtype=t.dtype)

def _randn(*args, **kw):
    import random as _r
    n = args[0] if args else 1
    if isinstance(n, tuple): n = n[0]
    return Tensor([_r.gauss(0, 1) for _ in range(n)], device=kw.get('device', 'cpu'))

def _rand(*args, **kw):
    import random as _r
    n = args[0] if args else 1
    if isinstance(n, tuple): n = n[0]
    return Tensor([_r.random() for _ in range(n)], device=kw.get('device', 'cpu'))

def _full(size, fill, **kw):
    if isinstance(size, tuple): n = size[0]
    else: n = size
    return Tensor([fill] * n, device=kw.get('device', 'cpu'))

def _full_like(t, fill):
    return Tensor([fill] * len(t._v), device=t.device, dtype=t.dtype)

def _empty_like(t, **kw):
    return Tensor([0.0] * len(t._v), device=t.device, dtype=t.dtype)

def _randperm(n, **kw):
    import random as _r
    vals = list(range(n))
    _r.shuffle(vals)
    return Tensor(vals, device=kw.get('device', 'cpu'))

def _randint(low, high, size, **kw):
    import random as _r
    if isinstance(size, tuple): n = size[0]
    else: n = size
    return Tensor([_r.randint(low, high - 1) for _ in range(n)], device=kw.get('device', 'cpu'), dtype=_int32)

def _linspace(start, end, steps, **kw):
    step = (end - start) / max(steps - 1, 1)
    return Tensor([start + i * step for i in range(steps)], device=kw.get('device', 'cpu'))


class _NoGradCtx:
    def __enter__(self): return None
    def __exit__(self, *a): pass
    def __call__(self, fn):
        def w(*a, **kw): return fn(*a, **kw)
        return w

def _no_grad():
    return _NoGradCtx()

def _save(obj, f):
    pass  # no-op

def _load(f, **kw):
    return {}

class _NNModule:
    def eval(self): pass
    def train(self): pass
    def zero_grad(self, set_to_none=True): pass
    def state_dict(self):
        return {'layer.weight': Tensor([0.1]*5), 'layer.bias': Tensor([0.01]*3)}
    def named_parameters(self):
        return [('layer.weight', _Param(Tensor([0.1]*5))),
                ('layer.bias', _Param(Tensor([0.01]*3)))]
    def __call__(self, x):
        return Tensor([0.5] * 10)

class _Param:
    def __init__(self, data): self.data = data; self.grad = None

# Build the torch module
_torch = types.ModuleType('torch')
_torch.Tensor = Tensor
_torch.tensor = _tensor
_torch.isfinite = _tensor_isfinite
_torch.norm = _norm
_torch.dot = _dot
_torch.mean = _torch_mean
_torch.std = _torch_std
_torch.quantile = _quantile
_torch.clamp = _clamp
_torch.relu = _relu
_torch.argmax = _argmax
_torch.sign = _sign
_torch.cat = _cat
_torch.stack = _stack
_torch.gather = _gather
_torch.empty = _empty
_torch.zeros = _zeros
_torch.ones = _ones
_torch.zeros_like = _zeros_like
_torch.randn_like = _randn_like
_torch.randn = _randn
_torch.rand = _rand
_torch.full = _full
_torch.full_like = _full_like
_torch.empty_like = _empty_like
_torch.randperm = _randperm
_torch.randint = _randint
_torch.linspace = _linspace
_torch.no_grad = _no_grad
_torch.save = _save
_torch.load = _load
_torch.int32 = _int32
_torch.float32 = _float32
_torch.float = _float32
_torch.all = _all
_torch.any = _any
_torch.sqrt = lambda t: Tensor([math.sqrt(max(x, 0)) for x in t._v]) if isinstance(t, Tensor) else math.sqrt(t)
_torch.nonzero = lambda t: t.nonzero() if isinstance(t, Tensor) else None

# nn sub-modules
_nn = types.ModuleType('torch.nn')
_nn.Module = _NNModule
_nn.functional = types.ModuleType('torch.nn.functional')
_torch.nn = _nn

# autograd
_autograd = types.ModuleType('torch.autograd')
_torch.autograd = _autograd

# distributions
_dist = types.ModuleType('torch.distributions')
_dist.Categorical = type('Categorical', (), {})
_torch.distributions = _dist

sys.modules['torch'] = _torch
sys.modules['torch.nn'] = _nn
sys.modules['torch.nn.functional'] = _nn.functional
sys.modules['torch.autograd'] = _autograd
sys.modules['torch.distributions'] = _dist

# Also patch common sub-imports
import torch
F_mock = types.ModuleType('torch.nn.functional')
F_mock.cross_entropy = lambda *a, **kw: Tensor([0.5])
F_mock.one_hot = lambda t, nc: Tensor([0]*nc)
F_mock.relu = _relu
sys.modules['torch.nn.functional'] = F_mock
_torch.nn.functional = F_mock

# Now import mos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import algorithms.attack.mos as mos

# ═══════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════
def vec(*vals):
    """Build a Tensor from values."""
    return Tensor(list(vals))

def assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg}: {a} != {b} (tol={tol})")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)

def assert_false(cond, msg=""):
    if cond:
        raise AssertionError(msg)

def assert_is_none(val, msg=""):
    if val is not None:
        raise AssertionError(f"{msg}: expected None, got {val}")

def assert_is_not_none(val, msg=""):
    if val is None:
        raise AssertionError(f"{msg}: expected not None")

def reset_stability_state():
    mos._LAST_VALID_GUIDANCE = None
    mos._LAST_VALID_BUDGET = None
    mos._BUDGET_EMA = None
    mos._LAST_BENIGN_MEAN_NORM = None
    mos._STABILITY_STATE_NUMEL = None


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_safe_normalize():
    """1. safe_normalize: basic validation"""
    # None
    v, n, ok = mos.safe_normalize(None)
    assert_is_none(v); assert_close(n, 0.0); assert_false(ok)

    # NaN
    v, n, ok = mos.safe_normalize(vec(1.0, float('nan'), 2.0))
    assert_is_none(v); assert_true(math.isnan(n)); assert_false(ok)

    # Inf → norm reported as NaN (caught by isfinite check before norm)
    v, n, ok = mos.safe_normalize(vec(1.0, float('inf')))
    assert_is_none(v); assert_true(math.isnan(n)); assert_false(ok)

    # Near-zero (< min_norm)
    v, n, ok = mos.safe_normalize(vec(1e-10, -1e-10), min_norm=1e-8)
    assert_is_none(v); assert_false(ok)

    # Exactly zero
    v, n, ok = mos.safe_normalize(vec(0.0, 0.0, 0.0), min_norm=1e-8)
    assert_is_none(v); assert_close(n, 0.0); assert_false(ok)

    # Valid → unit
    v, n, ok = mos.safe_normalize(vec(3.0, 4.0), min_norm=1e-8)
    assert_is_not_none(v); assert_close(n, 5.0); assert_true(ok)
    norm_out = math.sqrt(sum(x*x for x in v._v))
    assert_close(norm_out, 1.0, tol=1e-6)

    # numel mismatch
    v, n, ok = mos.safe_normalize(vec(1.0, 2.0), expected_numel=3)
    assert_is_none(v); assert_false(ok)

    # numel match
    v, n, ok = mos.safe_normalize(vec(1.0, 2.0), expected_numel=2)
    assert_is_not_none(v); assert_true(ok)


def test_no_zero_masking():
    """2. Zero vector NOT masked by eps"""
    v, n, ok = mos.safe_normalize(vec(0.0, 0.0), eps=1e-12, min_norm=1e-8)
    assert_is_none(v); assert_false(ok); assert_close(n, 0.0)


def test_ce_cw_both_zero():
    """3. CE/CW both zero → both invalid → fallback"""
    reset_stability_state()
    g_ce = vec(0.0, 0.0, 0.0)
    g_cw = vec(0.0, 0.0, 0.0)
    # 2 benign clients, 3 params each → shape (2, 3)
    benign_grads = Tensor([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], shape=(2, 3))
    benign_mean = vec(0.55, 1.1, 1.65)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=3, lam=0.5, min_norm=1e-8
    )
    assert_false(diag['ce_valid']); assert_false(diag['cw_valid'])
    src = diag['guidance_source']
    assert_true(src in ('benign_variance_fallback', 'none'), f"Unexpected source: {src}")
    if src == 'benign_variance_fallback':
        assert_is_not_none(guidance)
        gnorm = math.sqrt(sum(x*x for x in guidance._v))
        assert_true(gnorm > 0, "guidance norm should be > 0")


def test_ce_valid_cw_zero():
    """4. CE valid, CW zero → use CE only"""
    reset_stability_state()
    g_ce = vec(3.0, 4.0)    # norm=5
    g_cw = vec(0.0, 0.0)
    benign_grads = Tensor([0.1, 0.2, 0.15, 0.25], shape=(2, 2))
    benign_mean = vec(0.125, 0.225)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_true(diag['ce_valid']); assert_false(diag['cw_valid'])
    assert_true(diag['guidance_source'] == 'ce', f"Expected 'ce', got {diag['guidance_source']}")
    assert_is_not_none(guidance)
    gnorm = math.sqrt(sum(x*x for x in guidance._v))
    assert_close(gnorm, 1.0)


def test_cw_valid_ce_zero():
    """5. CW valid, CE zero → use CW only"""
    reset_stability_state()
    g_ce = vec(0.0, 0.0)
    g_cw = vec(6.0, 8.0)    # norm=10
    benign_grads = Tensor([0.1, 0.2, 0.15, 0.25], shape=(2, 2))
    benign_mean = vec(0.125, 0.225)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_false(diag['ce_valid']); assert_true(diag['cw_valid'])
    assert_true(diag['guidance_source'] == 'cw', f"Expected 'cw', got {diag['guidance_source']}")
    gnorm = math.sqrt(sum(x*x for x in guidance._v))
    assert_close(gnorm, 1.0)


def test_ce_cw_cancellation():
    """6. CE/CW exactly opposite → cancellation → conflict resolution → single direction"""
    reset_stability_state()
    g_ce = vec(1.0, 0.0, 0.0)           # unit along x
    g_cw = vec(-1.0, 0.0, 0.0)          # exactly opposite unit
    benign_grads = Tensor([0.0]*12, shape=(4, 3))
    benign_mean = vec(0.0, 0.0, 0.0)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=3, lam=0.5, min_norm=1e-8
    )
    assert_true(diag['ce_valid']); assert_true(diag['cw_valid'])
    cos = diag['ce_cw_cosine']
    assert_close(cos, -1.0)
    assert_true('conflict' in diag['guidance_source'],
                f"Expected conflict source, got {diag['guidance_source']}")
    assert_true(diag['guidance_fallback_used'])
    assert_is_not_none(guidance)
    gnorm = math.sqrt(sum(x*x for x in guidance._v))
    assert_close(gnorm, 1.0)


def test_historical_pop_fallback():
    """7. CE/CW invalid, historical_pop valid → use history direction"""
    reset_stability_state()
    g_ce = vec(0.0, 0.0, 0.0)
    g_cw = vec(0.0, 0.0, 0.0)
    historical_pop = vec(1.0, 2.0, 2.0)  # norm=3
    benign_grads = Tensor([0.1]*12, shape=(4, 3))
    benign_mean = vec(0.1, 0.1, 0.1)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, historical_pop, benign_grads, benign_mean,
        total_params=3, lam=0.5, min_norm=1e-8
    )
    assert_false(diag['ce_valid']); assert_false(diag['cw_valid'])
    assert_true(diag['historical_direction_valid'])
    assert_true(diag['guidance_source'] == 'historical_perturbation',
                f"Got {diag['guidance_source']}")
    assert_true(diag['guidance_fallback_used'])
    assert_is_not_none(guidance)


def test_benign_variance_fallback():
    """8. All invalid, no history → benign variance fallback"""
    reset_stability_state()
    g_ce = vec(0.0, 0.0)
    g_cw = vec(0.0, 0.0)
    # 2 clients, 2 params each → shape (2, 2)
    benign_grads = Tensor([10.0, -5.0, 0.1, 0.2], shape=(2, 2))
    benign_mean = vec(5.05, -2.4)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_false(diag['ce_valid']); assert_false(diag['cw_valid'])
    assert_false(diag['historical_direction_valid'])
    assert_true(diag['guidance_source'] == 'benign_variance_fallback',
                f"Got {diag['guidance_source']}")
    assert_is_not_none(guidance)


def test_cached_guidance_fallback():
    """9. Cached last valid guidance as fallback"""
    reset_stability_state()
    cached = vec(3.0, 4.0)   # real attack direction from previous round
    mos._LAST_VALID_GUIDANCE = cached

    g_ce = vec(0.0, 0.0)
    g_cw = vec(0.0, 0.0)
    # 4 clients, 2 params each → all zero
    benign_grads = Tensor([0.0]*8, shape=(4, 2))
    benign_mean = vec(0.0, 0.0)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_true(diag['guidance_source'] == 'cached_last_valid',
                f"Got {diag['guidance_source']}")
    assert_true(diag['guidance_fallback_used'])
    assert_is_not_none(guidance)


def test_guidance_cache_real_directions():
    """10. Only real attack directions cached, not fallbacks"""
    reset_stability_state()
    # Fallback to benign variance → NOT cached
    g_ce = vec(0.0, 0.0)
    g_cw = vec(0.0, 0.0)
    benign_grads = Tensor([10.0, 0.0, 0.0, 0.0], shape=(2, 2))
    benign_mean = vec(5.0, 0.0)

    guidance, diag = mos._construct_attack_guidance(
        g_ce, g_cw, None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_true(diag['guidance_source'] == 'benign_variance_fallback')
    assert_is_none(mos._LAST_VALID_GUIDANCE, "Fallback should NOT be cached")

    # Real CE/CW → cached
    reset_stability_state()
    guidance2, diag2 = mos._construct_attack_guidance(
        vec(3.0, 4.0), vec(4.0, 3.0), None, benign_grads, benign_mean,
        total_params=2, lam=0.5, min_norm=1e-8
    )
    assert_true(diag2['guidance_source'] == 'ce_cw_combined')
    assert_is_not_none(mos._LAST_VALID_GUIDANCE, "Real guidance SHOULD be cached")


def test_cache_dimension_reset():
    """11. Module cache resets on model dimension change"""
    reset_stability_state()
    mos._STABILITY_STATE_NUMEL = 5
    mos._LAST_VALID_GUIDANCE = vec(1.0, 0.0, 0.0, 0.0, 0.0)
    mos._LAST_VALID_BUDGET = 10.0
    mos._BUDGET_EMA = 10.0
    mos._LAST_BENIGN_MEAN_NORM = 5.0

    # Dimension change
    if mos._STABILITY_STATE_NUMEL is not None and mos._STABILITY_STATE_NUMEL != 8:
        mos._LAST_VALID_GUIDANCE = None
        mos._LAST_VALID_BUDGET = None
        mos._BUDGET_EMA = None
        mos._LAST_BENIGN_MEAN_NORM = None

    assert_is_none(mos._LAST_VALID_GUIDANCE)
    assert_is_none(mos._LAST_VALID_BUDGET)
    assert_is_none(mos._BUDGET_EMA)
    assert_is_none(mos._LAST_BENIGN_MEAN_NORM)


def test_budget_explosion():
    """12. Budget explosion capped at 2x"""
    reset_stability_state()
    raw = 100000.0
    bounded, ema = mos._bounded_budget(raw, previous_budget=100.0, previous_ema=100.0,
                                        beta=0.9, growth_cap=2.0, shrink_cap=0.25)
    assert_true(bounded <= 200.0 + 1e-9, f"bounded={bounded} > 200")
    assert_true(bounded <= raw)


def test_budget_shrink_floor():
    """13. Budget shrink floor at 0.25x"""
    reset_stability_state()
    bounded, ema = mos._bounded_budget(10.0, previous_budget=100.0, previous_ema=100.0,
                                        beta=0.9, growth_cap=2.0, shrink_cap=0.25)
    assert_true(bounded >= 25.0 - 1e-9, f"bounded={bounded} < 25")


def test_budget_first_round():
    """14. First round uses raw budget"""
    reset_stability_state()
    bounded, ema = mos._bounded_budget(50.0, previous_budget=None, previous_ema=None,
                                        beta=0.9, growth_cap=2.0, shrink_cap=0.25)
    assert_close(bounded, 50.0); assert_close(ema, 50.0)


def test_budget_anomaly_detection():
    """15. Budget anomaly logging logic"""
    previous_budget = 100.0
    previous_mean_norm = 10.0
    raw_budget = 1500.0
    benign_mean_norm = 120.0

    budget_growth_ratio = raw_budget / (previous_budget + 1e-12)
    mean_growth_ratio = benign_mean_norm / (previous_mean_norm + 1e-12)
    anomaly = budget_growth_ratio > 10.0 or mean_growth_ratio > 10.0

    assert_true(anomaly)
    assert_close(budget_growth_ratio, 15.0)
    assert_close(mean_growth_ratio, 12.0)


def test_historical_seed_rescaling():
    """16. Historical seed uses current bounded budget"""
    benign_mean = vec(1.0, 2.0, 3.0)
    hist_pert = vec(0.0, 0.0, 5.0)  # norm=5

    hist_unit, hist_norm, hist_valid = mos.safe_normalize(hist_pert, min_norm=1e-8, expected_numel=3)
    assert_true(hist_valid); assert_close(hist_norm, 5.0)
    # unit is [0, 0, 1]
    assert_close(hist_unit._v[2], 1.0, tol=1e-6)

    bounded_budget = 10.0; scale = 0.5
    seed_norm = scale * bounded_budget  # = 5.0
    assert_close(seed_norm, 5.0)


def test_tie_break_knee_selection():
    """17. Knee-point selection avoids always picking budget-extreme endpoint"""
    # 3 candidates: stealthy (1,0), attack (0,1), knee (0.5,0.5)
    pop_data = [0.0, 0.0, 5.0, 0.0, 2.5, 0.0]
    pop = Tensor(pop_data, shape=(3, 2))

    # R(x)=[1,0,0.5], A(x)=[0,5,2.5]  → obj = [-R, -A]
    obj_stealth = Tensor([1.0, 0.0, 0.5])
    obj_data = [-1.0, 0.0, -0.5, 0.0, -5.0, -2.5]
    objectives = Tensor(obj_data, shape=(2, 3))

    class Args:
        final_selection_mode = 'balanced_knee'
        selection_tie_tol = 1e-6
        final_stealth_weight = 0.5
        final_attack_weight = 0.5
        final_attack_floor_ratio = 0.0

    best_idx, diag = mos.select_final_solution(pop, objectives, constraint_scores=obj_stealth, args=Args())
    assert_true(diag['selection_mode'] == 'balanced_knee')
    # With knee point selection (ideal-point distance), idx 2 (knee) should win
    assert_true(best_idx == 2, f"Expected knee point idx=2, got {best_idx}")


def test_weighted_sum_selection():
    """18. weighted_sum mode with tie-break"""
    pop_data = [0.0, 0.0, 5.0, 0.0, 2.5, 0.0]
    pop = Tensor(pop_data)
    pop.shape = (3, 2)
    obj_stealth = Tensor([1.0, 0.0, 0.5])

    obj_data = [-1.0, 0.0, -0.5, 0.0, -5.0, -2.5]
    objectives = Tensor(obj_data)
    objectives.shape = (2, 3)

    class Args:
        final_selection_mode = 'weighted_sum'
        selection_tie_tol = 1e-6
        final_stealth_weight = 0.5
        final_attack_weight = 0.5
        final_attack_floor_ratio = 0.0

    best_idx, diag = mos.select_final_solution(pop, objectives, constraint_scores=obj_stealth, args=Args())
    assert_true(diag['selection_mode'] == 'weighted_sum')
    # 0.5/0.5 tie → tie-break: higher stealth → idx 0
    assert_true(best_idx == 0, f"Expected idx 0 (stealth in tie-break), got {best_idx}")


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    tests = [
        ("safe_normalize basic validation", test_safe_normalize),
        ("Zero vector not masked by eps", test_no_zero_masking),
        ("CE/CW both zero → invalid", test_ce_cw_both_zero),
        ("CE valid, CW zero → CE-only", test_ce_valid_cw_zero),
        ("CW valid, CE zero → CW-only", test_cw_valid_ce_zero),
        ("CE/CW cancellation → conflict resolution", test_ce_cw_cancellation),
        ("Historical perturbation fallback", test_historical_pop_fallback),
        ("Benign variance fallback", test_benign_variance_fallback),
        ("Cached guidance fallback", test_cached_guidance_fallback),
        ("Cache only stores real attack directions", test_guidance_cache_real_directions),
        ("Cache resets on dimension change", test_cache_dimension_reset),
        ("Budget explosion capped at 2x", test_budget_explosion),
        ("Budget shrink floor at 0.25x", test_budget_shrink_floor),
        ("First round uses raw budget", test_budget_first_round),
        ("Budget anomaly detection logic", test_budget_anomaly_detection),
        ("Historical seed rescaling", test_historical_seed_rescaling),
        ("Knee-point selection avoids budget extremes", test_tie_break_knee_selection),
        ("Weighted-sum mode with tie-break", test_weighted_sum_selection),
    ]

    failed = 0
    for name, fn in tests:
        try:
            print(f"  [{name}] ...", end=" ")
            fn()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {len(tests)-failed}/{len(tests)} passed")
    if failed:
        print(f"  {failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
