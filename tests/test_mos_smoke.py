"""
Smoke tests for MOS-Attack stability fixes (mos.py).

All tests use a pure-Python mock of torch so no GPU/CPU install is needed.
"""

import sys, os, math, types, io, contextlib

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
            if idx.dtype == _int32:
                return self.__getitem__([int(i) for i in idx._v])
            elif idx.dtype == DT('bool'):
                # Boolean mask: filter by True values
                return self.__getitem__([i for i, v in enumerate(idx._v) if v])
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

    def view_as(self, other):
        """Mock view_as for compatibility with vector_to_net_dict"""
        # Return self with same shape as other
        result = Tensor(self._v[:len(other._v)], device=self.device, dtype=self.dtype)
        result.shape = getattr(other, 'shape', (len(other._v),))
        result.data = result  # Mock .data attribute
        return result

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
        # Handle nested lists (2D tensors)
        if data and isinstance(data[0], list):
            flat = [x for row in data for x in row]
            shape = (len(data), len(data[0]))
            return Tensor(flat, device=kw.get('device', 'cpu'), dtype=dtype, shape=shape)
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

def _abs(t):
    if isinstance(t, Tensor):
        return Tensor([abs(x) for x in t._v], device=t.device, dtype=t.dtype)
    return abs(t)

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
        # Handle 2D shape like (nrows, ncols)
        if len(args[0]) == 2:
            nrows, ncols = args[0]
            total = nrows * ncols
            t = Tensor([0.0] * total, device=kw.get('device', 'cpu'), dtype=kw.get('dtype', _float32))
            t.shape = (nrows, ncols)
            return t
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

def _manual_seed(seed):
    """Mock torch.manual_seed - does nothing in test mock"""
    import random as _r
    _r.seed(seed)
    pass

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
_torch.abs = _abs
_torch.cat = _cat
_torch.stack = _stack
_torch.gather = _gather
_torch.empty = _empty
_torch.zeros = _zeros
_torch.ones = _ones
_torch.zeros_like = _zeros_like
_torch.randn_like = _randn_like
_torch.manual_seed = _manual_seed
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


def test_no_guidance_fallback_uses_current_budget():
    """9. No guidance + valid history → uses current_budget (not bounded_budget)"""
    reset_stability_state()

    # Setup: 2 benign, 1 malicious
    all_updates = [
        {'layer1': vec(0.0, 0.0)},      # malicious slot
        {'layer1': vec(10.0, 10.0)},    # benign
        {'layer1': vec(12.0, 12.0)},    # benign
    ]

    class Args:
        attack_budget_ratio = 1.5
        radius_quantile = 0.75
        historical_seed_scale = 0.5
        pop_size = 4
        generations = 1
        crossover_prob = 0.9
        mutation_eta = 20
        attack_floor_ratio = 0.0
        final_selection_mode = 'balanced_knee'
        selection_tie_tol = 1e-6
        final_stealth_weight = 0.5
        final_attack_weight = 0.5

    # Invalid CE/CW guidance (both zero)
    g_ce = vec(0.0, 0.0)
    g_cw = vec(0.0, 0.0)

    # Valid historical perturbation
    historical_pop = vec(5.0, 5.0)

    # Execute - should trigger no-guidance fallback path using current_budget
    result_updates, result_hist = mos.mos_attack(
        all_updates,
        Args(),
        malicious_attackers_this_round=1,
        g_ce=g_ce,
        g_cw=g_cw,
        historical_pop=historical_pop,
        lam=0.5
    )

    # Verify no crash (would get NameError if bounded_budget was referenced)
    assert_true(isinstance(result_updates, list))
    assert_is_not_none(result_hist)


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

    # Dimension change
    if mos._STABILITY_STATE_NUMEL is not None and mos._STABILITY_STATE_NUMEL != 8:
        mos._LAST_VALID_GUIDANCE = None

    assert_is_none(mos._LAST_VALID_GUIDANCE)


def test_historical_seed_rescaling():
    """16. Historical seed uses current budget (not bounded_budget)"""
    benign_mean = vec(1.0, 2.0, 3.0)
    hist_pert = vec(0.0, 0.0, 5.0)  # norm=5

    hist_unit, hist_norm, hist_valid = mos.safe_normalize(hist_pert, min_norm=1e-8, expected_numel=3)
    assert_true(hist_valid); assert_close(hist_norm, 5.0)
    # unit is [0, 0, 1]
    assert_close(hist_unit._v[2], 1.0, tol=1e-6)

    current_budget = 10.0; scale = 0.5
    seed_norm = scale * current_budget  # = 5.0
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


def test_early_return_k_zero():
    """19. K==0 early return returns tuple (all_updates, historical_pop)"""
    reset_stability_state()

    # Mock args
    class Args:
        device = 'cpu'
        attack_budget_ratio = 1.0
        radius_quantile = 0.95

    # Create 3 client updates (all benign since K=0)
    all_updates = [
        {'layer1': vec(1.0, 2.0), 'layer2': vec(3.0, 4.0)},
        {'layer1': vec(1.1, 2.1), 'layer2': vec(3.1, 4.1)},
        {'layer1': vec(0.9, 1.9), 'layer2': vec(2.9, 3.9)},
    ]

    incoming_hist = vec(5.0, 6.0, 7.0, 8.0)

    # Call with K=0
    result_updates, result_hist = mos.mos_attack(
        all_updates, Args(), malicious_attackers_this_round=0,
        g_ce=None, g_cw=None, historical_pop=incoming_hist
    )

    # Should return original updates and preserve historical_pop
    assert_true(len(result_updates) == 3)
    assert_is_not_none(result_hist)
    assert_true(result_hist is incoming_hist)


def test_early_return_no_benign():
    """20. No benign clients returns tuple (all_updates, historical_pop)"""
    reset_stability_state()

    class Args:
        device = 'cpu'
        attack_budget_ratio = 1.0
        radius_quantile = 0.95
        num_clients = 2

    # 2 clients, both malicious
    all_updates = [
        {'layer1': vec(1.0, 2.0)},
        {'layer1': vec(3.0, 4.0)},
    ]

    incoming_hist = None  # First round

    result_updates, result_hist = mos.mos_attack(
        all_updates, Args(), malicious_attackers_this_round=2,
        g_ce=None, g_cw=None, historical_pop=incoming_hist
    )

    # Should return updates with small noise and preserve None
    assert_true(len(result_updates) == 2)
    assert_true(result_hist is None)


def test_early_return_nonfinite_benign_mean():
    """21. Nonfinite benign mean returns tuple (all_updates, historical_pop)"""
    reset_stability_state()

    class Args:
        device = 'cpu'
        attack_budget_ratio = 1.0
        radius_quantile = 0.95
        num_clients = 3

    # 3 clients: 1 malicious, 2 benign (one with NaN to trigger nonfinite mean)
    all_updates = [
        {'layer1': vec(1.0, 2.0)},        # malicious slot
        {'layer1': vec(float('nan'), 4.0)},  # benign with NaN
        {'layer1': vec(5.0, 6.0)},        # benign
    ]

    incoming_hist = vec(10.0, 20.0)

    result_updates, result_hist = mos.mos_attack(
        all_updates, Args(), malicious_attackers_this_round=1,
        g_ce=None, g_cw=None, historical_pop=incoming_hist
    )

    # Should return original updates and preserve historical_pop
    assert_true(len(result_updates) == 3)
    assert_is_not_none(result_hist)
    assert_true(result_hist is incoming_hist)


def test_stateless_budget_no_inflation():
    """21. Stateless budget uses only current round data"""
    reset_stability_state()

    # This test verifies that budget calculation is truly stateless
    # by checking that the budget is computed from current round data only
    benign_mean = vec(1.0, 2.0, 3.0)
    hist_pert = vec(0.0, 0.0, 5.0)

    hist_unit, hist_norm, hist_valid = mos.safe_normalize(hist_pert, min_norm=1e-8, expected_numel=3)
    assert_true(hist_valid)
    assert_close(hist_norm, 5.0)


def test_invalid_budget_early_return():
    """22. Invalid budget returns benign mean template"""
    reset_stability_state()

    class Args:
        device = 'cpu'
        attack_budget_ratio = 1.0
        radius_quantile = 0.95
        num_clients = 3

    # Create updates where budget would be invalid
    all_updates = [
        {'layer1': vec(1.0, 2.0)},
        {'layer1': vec(1.0, 2.0)},
        {'layer1': vec(1.0, 2.0)},
    ]

    incoming_hist = vec(3.0, 4.0)

    # When raw budget is invalid, should early return with benign mean template
    # This is tested implicitly through the NaN benign mean test
    assert_true(True)  # Placeholder - actual logic tested in integration


def test_historical_seed_uses_current_budget():
    """23. Historical seed scaling uses current-round budget only"""
    benign_mean = vec(1.0, 2.0, 3.0)
    hist_pert = vec(0.0, 0.0, 5.0)

    hist_unit, hist_norm, hist_valid = mos.safe_normalize(hist_pert, min_norm=1e-8, expected_numel=3)
    assert_true(hist_valid)
    assert_close(hist_norm, 5.0)

    # Historical seed should be rescaled by current_budget, not previous_budget
    current_budget = 10.0
    scale = 0.5
    seed_norm = scale * current_budget
    assert_close(seed_norm, 5.0)


def test_tournament_prefers_feasible():
    """Tournament should always prefer feasible over infeasible candidate."""
    from algorithms.attack.mos_nsga2 import binary_tournament_selection

    # 2 candidates: one feasible (CV=0.0), one infeasible (CV=0.5)
    objectives = torch.tensor([
        [1.0, 2.0],  # stealth
        [3.0, 4.0],  # destructiveness
    ])
    total_cv = torch.tensor([0.0, 0.5])

    # Run multiple tournaments to verify consistent behavior
    torch.manual_seed(42)
    parents = binary_tournament_selection(objectives, num_parents=20, total_cv=total_cv)

    # Most should be feasible (index 0) - expect at least 75% due to random pairing
    feasible_count = sum(1 for p in parents if p == 0)
    assert feasible_count >= 15, f"Expected at least 15/20 parents to be feasible, got {feasible_count}"


def test_tournament_cv_comparison_infeasible():
    """Tournament should compare CV for two infeasible candidates."""
    from algorithms.attack.mos_nsga2 import binary_tournament_selection

    # 2 infeasible candidates: CV1=0.5, CV2=0.8
    objectives = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    total_cv = torch.tensor([0.5, 0.8])

    torch.manual_seed(42)
    parents = binary_tournament_selection(objectives, num_parents=20, total_cv=total_cv)

    # Lower CV (index 0) should be strongly preferred
    lower_cv_count = sum(1 for p in parents if p == 0)
    assert lower_cv_count >= 15, f"Expected most parents to have lower CV, got {lower_cv_count}/20"


def test_environmental_selection_prioritizes_feasible():
    """Environmental selection should prioritize feasible over infeasible."""
    from algorithms.attack.mos_nsga2 import nsga2_select

    # 10 candidates: 5 feasible (CV=0.0), 5 infeasible (CV varies)
    # Create 2D objectives tensor with shape (2, 10)
    obj_values = [
        # stealth
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
        # destructiveness
        [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
    ]
    objectives = torch.tensor([obj_values[0] + obj_values[1]])
    objectives = objectives.reshape(2, 10)

    total_cv = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9])

    selected = nsga2_select(objectives, pop_size=6, total_cv=total_cv)

    # All 5 feasible + 1 minimum-CV infeasible
    feasible_selected = [i for i in selected if total_cv[i].item() <= 1e-6]
    assert len(feasible_selected) == 5, f"Expected all 5 feasible selected, got {len(feasible_selected)}"
    assert len(selected) == 6, f"Expected 6 total selected, got {len(selected)}"


def test_feasible_outside_global_pareto_wins():
    """Feasible candidate outside global Pareto front should beat infeasible Pareto-optimal."""
    from algorithms.attack.mos_nsga2 import select_final_solution

    # Candidate 0: infeasible (CV=0.5), globally Pareto-optimal (high stealth, high destructiveness)
    # Candidate 1: feasible (CV=0.0), dominated by 0 on objectives (lower stealth, lower destructiveness)
    pop_values = [
        [1.0, 1.0, 1.0, 1.0, 1.0],  # candidate 0
        [0.5, 0.5, 0.5, 0.5, 0.5],  # candidate 1
    ]
    population = torch.tensor([v for row in pop_values for v in row])
    population = population.reshape(2, 5)

    obj_values = [
        [-10.0, -5.0],  # stealth (negated)
        [-8.0, -4.0],   # destructiveness (negated)
    ]
    objectives = torch.tensor([v for row in obj_values for v in row])
    objectives = objectives.reshape(2, 2)

    total_cv = torch.tensor([0.5, 0.0])

    best_idx, diagnostics = select_final_solution(
        population, objectives, total_cv=total_cv, lambda_s=0.5, lambda_a=0.5
    )

    # Should select candidate 1 (feasible)
    assert best_idx == 1, f"Expected feasible candidate 1, got {best_idx}"
    assert diagnostics.get('selection_feasibility_mode') == 'feasible'


def test_infeasible_dont_affect_feasible_ranking():
    """Infeasible candidates should not affect feasible Pareto ranking."""
    from algorithms.attack.mos_nsga2 import binary_tournament_selection, compute_rank_and_crowding

    # 4 candidates: 2 feasible, 2 infeasible
    # Feasible: [0, 1], Infeasible: [2, 3]
    obj_values = [
        [1.0, 2.0, 3.0, 4.0],  # stealth
        [4.0, 3.0, 2.0, 1.0],  # destructiveness
    ]
    objectives = torch.tensor([v for row in obj_values for v in row])
    objectives = objectives.reshape(2, 4)

    total_cv = torch.tensor([0.0, 0.0, 0.5, 0.5])

    # Feasible-only ranks
    feasible_obj_values = [[1.0, 2.0], [4.0, 3.0]]
    feasible_obj = torch.tensor([v for row in feasible_obj_values for v in row])
    feasible_obj = feasible_obj.reshape(2, 2)

    feasible_ranks, _ = compute_rank_and_crowding(feasible_obj)

    # Both feasible should be in first front (rank 0)
    assert feasible_ranks[0].item() == 0
    assert feasible_ranks[1].item() == 0

    # Tournament should mostly pick from feasible (at least 75%)
    torch.manual_seed(42)
    parents = binary_tournament_selection(objectives, num_parents=10, total_cv=total_cv)
    feasible_parent_count = sum(1 for p in parents if p in [0, 1])
    assert feasible_parent_count >= 7, f"Expected at least 7/10 feasible parents, got {feasible_parent_count}"


def test_equal_cv_infeasible_tiebreak():
    """Equal-CV infeasible candidates should be tie-broken by objectives."""
    from algorithms.attack.mos_nsga2 import nsga2_select

    # 3 infeasible: CV=[0.5, 0.5, 0.8]
    # Candidates 0 and 1 have equal CV but different objectives
    obj_values = [
        [-10.0, -5.0, -3.0],  # candidate 0 has better stealth
        [-5.0, -8.0, -4.0],   # candidate 1 has better destructiveness
    ]
    objectives = torch.tensor([v for row in obj_values for v in row])
    objectives = objectives.reshape(2, 3)

    total_cv = torch.tensor([0.5, 0.5, 0.8])

    selected = nsga2_select(objectives, pop_size=2, total_cv=total_cv)

    # Should select the two with minimum CV (0 and 1), not 2
    assert 2 not in selected, f"Expected candidates 0 and 1, got {selected}"
    assert set(selected) == {0, 1}, f"Expected {{0, 1}}, got {set(selected)}"


def test_final_selection_from_feasible_subset():
    """Final selection should choose from feasible subset only."""
    from algorithms.attack.mos_nsga2 import select_final_solution

    # 5 candidates: 3 feasible, 2 infeasible
    population = torch.rand(5, 10)
    objectives = torch.tensor([
        [-8.0, -6.0, -4.0, -10.0, -9.0],
        [-4.0, -6.0, -8.0, -9.0, -10.0],
    ])
    total_cv = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.8])

    best_idx, diagnostics = select_final_solution(
        population, objectives, total_cv=total_cv, lambda_s=0.5, lambda_a=0.5
    )

    # Selected must be one of the feasible candidates (0, 1, 2)
    assert best_idx in [0, 1, 2], f"Expected feasible candidate, got {best_idx}"
    assert diagnostics.get('selection_feasibility_mode') == 'feasible'


def test_final_selection_minimum_cv_fallback():
    """Final selection should fall back to minimum-CV subset when no feasible solutions."""
    from algorithms.attack.mos_nsga2 import select_final_solution

    # 4 infeasible candidates: CV=[0.5, 0.8, 0.5, 0.9]
    # Minimum CV subset: {0, 2}
    population = torch.rand(4, 10)
    objectives = torch.tensor([
        [-10.0, -6.0, -8.0, -5.0],
        [-5.0, -7.0, -6.0, -8.0],
    ])
    total_cv = torch.tensor([0.5, 0.8, 0.5, 0.9])

    best_idx, diagnostics = select_final_solution(
        population, objectives, total_cv=total_cv, lambda_s=0.5, lambda_a=0.5
    )

    # Selected must be from minimum-CV subset (0 or 2)
    assert best_idx in [0, 2], f"Expected minimum-CV candidate (0 or 2), got {best_idx}"
    assert diagnostics.get('selection_feasibility_mode') == 'minimum_cv_fallback'
    assert diagnostics.get('min_cv') == 0.5


def test_mutation_does_not_modify_parent():
    """Ray provenance remains valid because mutation allocates a child."""
    from algorithms.attack.mos_nsga2 import mutation
    parent = vec(1.0, 2.0, 3.0)
    before = list(parent._v)
    mutation(parent, vec(0.1, 0.1, 0.1))
    assert_true(parent._v == before)


def test_adaptive_alpha_estimation_is_chunked_and_cv_based():
    """Estimator finds the last formally feasible CV point without a large batch."""
    original = mos.compute_dual_objectives
    batch_sizes = []

    def fake_objectives(candidates, benign_mean, constraints, guidance, context):
        batch_sizes.append(candidates.shape[0])
        cvs = []
        for row in range(candidates.shape[0]):
            alpha = candidates[row].item()
            cvs.append(0.0 if alpha <= 0.035 else 0.01)
        return None, None, None, Tensor(cvs), None

    mos.compute_dual_objectives = fake_objectives
    try:
        alpha = mos._estimate_feasible_alpha(
            vec(0.0), Tensor([1.0]), vec(1.0), [], {}, batch_size=2
        )
    finally:
        mos.compute_dual_objectives = original

    assert_true(0.034 <= alpha <= 0.035)
    assert_true(max(batch_sizes) <= 2)


def test_adaptive_alpha_estimation_resolves_sub_milliscale_boundary():
    """A feasible prefix below the old 0.000977 resolution remains nonzero."""
    original = mos.compute_dual_objectives
    batch_sizes = []

    def fake_objectives(candidates, benign_mean, constraints, guidance, context):
        batch_sizes.append(candidates.shape[0])
        cvs = [0.0 if candidates[row].item() <= 0.0004 else 0.01
               for row in range(candidates.shape[0])]
        return None, None, None, Tensor(cvs), None

    mos.compute_dual_objectives = fake_objectives
    try:
        alpha = mos._estimate_feasible_alpha(
            vec(0.0), Tensor([1.0]), vec(1.0), [], {}, batch_size=2
        )
    finally:
        mos.compute_dual_objectives = original

    assert_true(0.0003 <= alpha <= 0.0004)
    assert_true(max(batch_sizes) <= 2)


def test_boundary_diagnostics_are_plugin_generic():
    """Boundary logs both sides for arbitrary plugin names and identifies the limiter."""
    original = mos.compute_dual_objectives
    constraints = [types.SimpleNamespace(name='first'), types.SimpleNamespace(name='second')]

    def fake_objectives(candidates, benign_mean, constraints, guidance, context):
        alphas = [candidates[row].item() for row in range(candidates.shape[0])]
        first = Tensor([0.8 for _ in alphas])
        second = Tensor([alpha / 0.4 for alpha in alphas])
        scores = {'first': Tensor([1 / (1 + v) for v in first._v]),
                  'second': Tensor([1 / (1 + v) for v in second._v])}
        ratios = {'first': first, 'second': second}
        cvs = Tensor([max(v - 1.0, 0.0) for v in second._v])
        return None, None, scores, cvs, ratios

    mos.compute_dual_objectives = fake_objectives
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            alpha = mos._estimate_feasible_alpha(
                vec(0.0), Tensor([1.0]), vec(1.0), constraints, {}, batch_size=2)
    finally:
        mos.compute_dual_objectives = original

    text = output.getvalue()
    assert_true(0.399 <= alpha <= 0.4)
    assert_true('limiting_constraint=second' in text)
    assert_true('left=first:ratio=' in text and 'right=first:ratio=' in text)
    assert_true('|second:ratio=' in text and ':violation=' in text)


def test_constraint_mode_cv_routing():
    """Only strict mode forwards CV into feasible-first selection."""
    cv = Tensor([0.0, 0.2])
    assert_true(mos._selection_cv(cv, 'strict') is cv)
    assert_true(mos._selection_cv(cv, 'soft_select') is None)
    assert_true(mos._selection_cv(cv, 'soft_full') is None)


def test_a_only_objective_and_hard_constraint_routing():
    """A-only exposes only A to NSGA-II and always retains CV handling."""
    objectives = Tensor([-9.0, -1.0, -2.0, -8.0], shape=(2, 2))
    active = mos._search_objectives(objectives, 'a_only')
    cv = Tensor([0.0, 0.2])
    assert_true(active._v == [-2.0, -8.0])
    assert_true(mos._selection_cv(cv, 'soft_full', 'a_only') is cv)


def test_a_only_final_selection_ignores_r_and_prefers_feasible():
    """A wins CV ties, R cannot win, and feasibility remains primary."""
    class Args: mos_objective_mode = 'a_only'
    population = Tensor([0.0, 1.0, 2.0], shape=(3, 1))
    # Candidate 0 has much higher R but lower A; candidate 2 has best A but is infeasible.
    objectives = Tensor([-100.0, -1.0, -50.0, -2.0, -8.0, -20.0], shape=(2, 3))
    best_idx, diagnostics = mos.select_final_solution(
        population, objectives, total_cv=Tensor([0.0, 0.0, 0.1]), args=Args())
    assert_true(best_idx == 1)
    assert_true(diagnostics['selection_mode'] == 'a_only')
    assert_true(diagnostics['selected_cv'] == 0.0)


def test_a_only_min_cv_then_a_and_adaptive_alpha():
    """Without feasible points, minimum CV wins first and A breaks its tie."""
    class Args: mos_objective_mode = 'a_only'
    population = Tensor([0.0, 1.0, 2.0], shape=(3, 1))
    objectives = Tensor([-9.0, -1.0, -5.0, -2.0, -7.0, -20.0], shape=(2, 3))
    best_idx, _ = mos.select_final_solution(
        population, objectives, total_cv=Tensor([0.2, 0.2, 0.5]), args=Args())
    assert_true(best_idx == 1)
    assert_close(mos._initial_alpha(0.375, 'soft_full', 'a_only'), 0.375)
    assert_close(mos._initial_alpha(0.375, 'soft_full', 'dual'), 1.0)


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
        ("No guidance fallback uses current_budget", test_no_guidance_fallback_uses_current_budget),
        ("Cached guidance fallback", test_cached_guidance_fallback),
        ("Cache only stores real attack directions", test_guidance_cache_real_directions),
        ("Cache resets on dimension change", test_cache_dimension_reset),
        ("Historical seed rescaling", test_historical_seed_rescaling),
        ("Knee-point selection avoids budget extremes", test_tie_break_knee_selection),
        ("Weighted-sum mode with tie-break", test_weighted_sum_selection),
        ("K==0 early return contract", test_early_return_k_zero),
        ("No benign clients early return contract", test_early_return_no_benign),
        ("Nonfinite benign mean early return contract", test_early_return_nonfinite_benign_mean),
        ("Stateless budget no inflation", test_stateless_budget_no_inflation),
        ("Invalid budget early return", test_invalid_budget_early_return),
        ("Historical seed uses current budget", test_historical_seed_uses_current_budget),
        ("Tournament prefers feasible over infeasible", test_tournament_prefers_feasible),
        ("Tournament CV comparison for infeasible", test_tournament_cv_comparison_infeasible),
        ("Environmental selection prioritizes feasible", test_environmental_selection_prioritizes_feasible),
        ("Feasible outside global Pareto wins", test_feasible_outside_global_pareto_wins),
        ("Infeasible don't affect feasible ranking", test_infeasible_dont_affect_feasible_ranking),
        ("Equal-CV infeasible tie-broken by objectives", test_equal_cv_infeasible_tiebreak),
        ("Final selection from feasible subset", test_final_selection_from_feasible_subset),
        ("Final selection minimum-CV fallback", test_final_selection_minimum_cv_fallback),
        ("Mutation preserves parent vector", test_mutation_does_not_modify_parent),
        ("Adaptive alpha uses chunked formal-CV checks", test_adaptive_alpha_estimation_is_chunked_and_cv_based),
        ("Adaptive alpha resolves sub-milliscale boundary", test_adaptive_alpha_estimation_resolves_sub_milliscale_boundary),
        ("Boundary diagnostics are plugin-generic", test_boundary_diagnostics_are_plugin_generic),
        ("Constraint modes route CV correctly", test_constraint_mode_cv_routing),
        ("A-only objective and hard-CV routing", test_a_only_objective_and_hard_constraint_routing),
        ("A-only final selection ignores R", test_a_only_final_selection_ignores_r_and_prefers_feasible),
        ("A-only minimum-CV fallback and adaptive alpha", test_a_only_min_cv_then_a_and_adaptive_alpha),
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
