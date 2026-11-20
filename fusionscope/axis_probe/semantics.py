"""Check compiled counts directly, against exported sites, and on threshold inputs."""
import numpy as np

from .design import VARIANTS, function


def pressure_inputs(case, params):
    """Two dynamic affine/dense parameter sets, also used by the reduced graphs."""
    import jax.numpy as jnp
    if case.family not in ("E", "M"):
        return {}
    width = case.shape[-1]
    eps = np.finfo(np.float32).eps
    tiny = np.finfo(np.float32).tiny
    stress = {}
    for label, values, w, b in (
        ("signed_zero_and_normal", [0., -0., tiny, -tiny, 1., -1.], 1., 0.),
        ("producer_cancellation", [1.+eps, 1., 1.-eps], 1.+eps, -(1.+2.*eps)),
    ):
        x = np.resize(np.asarray(values, np.float32), case.shape)
        p = list(params)
        for i in (0, 2):
            kernel = np.full(width, w if i == 0 else 1., np.float32)
            p[i] = jnp.asarray(np.diag(kernel) if case.family == "M" else kernel)
            p[i+1] = jnp.full(width, b if i == 0 else 0., jnp.float32)
        stress[label] = (jnp.asarray(x), tuple(p))
    return stress


def validate(case, executables, inputs):
    import jax
    from fusionscope.cpu_cost.workloads import oracle

    diag = jax.jit(function(case, "diagnostic"))
    rows = []
    for seed, args in inputs.items():
        _, sites = diag(*args)
        arrays = tuple(map(np.asarray, sites))
        refs = {q: [value for value, _ in oracle(q, arrays)] for q in ("count", "row")}
        results = {key: tuple(map(np.asarray, fn(*args))) for key, fn in executables.items()}
        main = results["V0", "on"][0]
        for (variant, setting), output in results.items():
            request = variant.split("_")[0]
            expected = refs.get(request, [])
            errors = [int(np.abs(v.astype(np.int64)-target).max())
                      for v, target in zip(output[1:], expected, strict=True)]
            counterpart = results.get((request+"_S", setting), output)
            pair_error = max((int(np.abs(v.astype(np.int64)-s.astype(np.int64)).max())
                              for v, s in zip(output[1:], counterpart[1:], strict=True)), default=0)
            main_ok = (output[0].dtype == main.dtype == np.float32 and output[0].shape == main.shape
                       and np.isfinite(output[0]).all() and np.allclose(output[0], main, rtol=3e-4, atol=3e-4))
            types_ok = all(v.dtype == np.int32 and v.shape == ref.shape
                           for v, ref in zip(output[1:], expected, strict=True))
            rows.append(dict(graph=case.name, variant=variant, setting=setting, input=seed,
                             main_ok=bool(main_ok), max_main_abs=float(np.abs(output[0]-main).max()),
                             counts_typed=types_ok, diagnostic_error=max(errors, default=0),
                             separate_error=pair_error,
                             valid=bool(main_ok and types_ok and not any(errors) and pair_error == 0)))
    return rows


def fixed_sites():
    import jax
    import jax.numpy as jnp
    from fusionscope.cpu_cost.workloads import observe, oracle

    tiny = np.finfo(np.float32).tiny
    sub = np.nextafter(np.float32(0), np.float32(1))
    categories = {"normal_and_signed_zero": [-tiny, -0., 0., tiny, -1., 1.],
                  "zero_nextafter": [-sub, -0., 0., sub],
                  "nonfinite": [-np.inf, -0., np.inf, np.nan]}
    rows = []
    for label, values in categories.items():
        a = np.array([values, list(reversed(values))], np.float32)
        b = a[:, ::-1].copy()
        for variant in VARIANTS[1:]:
            request, schedule = variant.split("_")
            compiled = jax.jit(lambda x, y: observe((x, y), request, schedule == "J"))
            actual = [np.asarray(v) for v in compiled(jnp.asarray(a), jnp.asarray(b))]
            expected = [v for v, _ in oracle(request, (a, b))]
            rows.append(dict(input=label, variant=variant, input_bits=a.view(np.uint32).tolist(),
                             actual=[v.tolist() for v in actual], ieee_reference=[v.tolist() for v in expected],
                             exact=all(v.dtype == np.int32 and np.array_equal(v, ref)
                                       for v, ref in zip(actual, expected, strict=True))))
    return rows
