"""Independent NumPy implementations for small synthetic fixtures."""
import jax
import numpy as np

from .common import MODES, dump, sites
from .graphs import program


def numpy_layer(family, x, w, b):
    if family == "M":
        return np.einsum("ij,jk->ik", x, w) + b
    if family == "C":
        padded = np.pad(x, ((0,0),(1,1),(1,1),(0,0)))
        out = np.zeros_like(x)
        for i in range(3):
            for j in range(3):
                out += np.einsum("bhwc,co->bhwo", padded[:,i:i+x.shape[1],j:j+x.shape[2]], w[i,j])
        return out + b
    if family == "N":
        rms = np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + np.float32(1e-4))
        x = x/rms
    return x*w+b


def check(root):
    rng = np.random.default_rng(507)
    zero = np.array([-2.0,-0.0,0.0,3.0,5.0], np.float32)
    assert int(np.sum(zero>0)) == int(np.sum(np.maximum(zero,0)>0)) == 2
    assert np.min(zero) == -2 and np.min(np.maximum(zero,0)) == 0
    rows = []
    for fam in "EMNC":
        shape = (1,4,4,4) if fam == "C" else (4,4)
        x = rng.integers(-4,5, size=shape).astype(np.float32)/4
        x.flat[:4] = [-1,-0.,0.,1]
        ws = {"M": (4,4), "C": (3,3,4,4)}.get(fam, (4,))
        p = tuple(rng.integers(-4,5,size=s).astype(np.float32)/4 for s in (ws,(4,),ws,(4,)))
        z = numpy_layer(fam, x, *p[:2])
        a = np.maximum(z,0)
        z2 = numpy_layer(fam, a, *p[2:])
        if fam == "C":
            z2 += x
        counts = dict(u=int((z>0).sum()), v=int((z2>0).sum()))
        for top in ("chain","branch"):
            y = np.maximum(z2,0)*np.float32(0.75)+np.float32(0.125)
            if top == "branch":
                y += a*np.float32(0.25)
            for mode in MODES:
                out = jax.jit(program(f"{fam}0_{top}", mode))(x,p)
                main, *q = out if isinstance(out, tuple) else (out,)
                assert np.allclose(main, y, rtol=1e-5, atol=2e-6), (fam,top,mode)
                assert [int(v) for v in q] == [counts[k] for k in sites(mode)], (fam,top,mode)
                rows.append({"family":fam,"topology":top,"mode":mode,"counts":[int(v) for v in q]})
    dump(root / "validation/numpy.json", {"signed_zero_positive_count":2,"pre_min":-2,"post_min":0,"checks":rows})
