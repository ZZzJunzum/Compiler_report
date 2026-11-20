"""Audit packed integer carries and the backend's predicate behavior on fixed arrays."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from fusionscope.common import ROOT,dump,read
from fusionscope.partition.workloads import packed_counts
from fusionscope.partition.policies import eligible

PATH = ROOT/"results/partition_boundary/contract.json"


def audit():
    results = []
    for width in (1,15,16,255,256,65535):
        col = np.arange(width)
        ones = np.ones(width,np.float32)
        a = np.stack((ones,np.where(col%2,1,-1),-ones)).astype(np.float32)
        b = np.stack((ones,np.where(col%3,1,-1),ones)).astype(np.float32)
        got = [np.asarray(x) for x in jax.jit(packed_counts)(jnp.asarray(a),jnp.asarray(b))]
        expected = [np.count_nonzero(q>0,axis=-1) for q in (a,b)]
        assert eligible(width) and all(x.dtype==np.int32 and np.array_equal(x,e)
                                      for x,e in zip(got,expected,strict=True))
        results.append(dict(width=width,count1=got[0].tolist(),count2=got[1].tolist()))
    rejected = []
    for w,v in ((0,0),(65536,65536),(8,7)):
        try:
            jax.eval_shape(packed_counts,jax.ShapeDtypeStruct((1,w),jnp.float32),
                           jax.ShapeDtypeStruct((1,v),jnp.float32))
        except ValueError as exc:
            rejected.append(dict(widths=[w,v],error=str(exc)))
        else:
            raise AssertionError((w,v))
    assert not eligible(0) and not eligible(65536)
    tiny = np.finfo(np.float32).tiny
    least = np.nextafter(np.float32(0),np.float32(1))
    x = np.array([[0.,-0.,tiny,-tiny,np.inf,-np.inf,np.nan,1.],
                  [least,least,-least,-least,0.,-0.,0.,-0.]],np.float32)
    arr = jnp.asarray(x)
    native = np.asarray(jax.jit(lambda v:jnp.sum(v>0,axis=-1,dtype=jnp.int32))(arr))
    packed = [np.asarray(v).tolist() for v in jax.jit(packed_counts)(arr,arr)]
    reference = np.count_nonzero(x>0,axis=-1).tolist()
    assert packed==[native.tolist(),native.tolist()] and packed[0][0]==reference[0]
    return dict(carries=results,rejections=rejected,edge_rows=["signed zeros, normal extremes, infinity, NaN",
                "two minimum positive FP32 subnormals"],host_ieee=reference,
                compiled_native=native.tolist(),compiled_packed=packed)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify",action="store_true")
    args = ap.parse_args()
    result = audit()
    if args.verify:
        assert result==read(PATH)
    else:
        dump(PATH,result)
    print("Checked six widths, carry saturation, rejection and fixed predicate edge cases.")
