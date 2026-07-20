import numpy as np
import sympy
import torch

# Each entry: (torch callable, numpy callable, sympy callable). All three take
# a single argument and are used to fit c * f(a*x + b) + d against a learned
# edge's activation curve.
SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x, lambda x: x),
    "x^2": (lambda x: x**2, lambda x: x**2, lambda x: x**2),
    "x^3": (lambda x: x**3, lambda x: x**3, lambda x: x**3),
    "x^4": (lambda x: x**4, lambda x: x**4, lambda x: x**4),
    "exp": (torch.exp, np.exp, sympy.exp),
    "log": (
        lambda x: torch.log(torch.abs(x) + 1e-4),
        lambda x: np.log(np.abs(x) + 1e-4),
        lambda x: sympy.log(sympy.Abs(x) + 1e-4),
    ),
    "sqrt": (
        lambda x: torch.sqrt(torch.abs(x)),
        lambda x: np.sqrt(np.abs(x)),
        lambda x: sympy.sqrt(sympy.Abs(x)),
    ),
    "tanh": (torch.tanh, np.tanh, sympy.tanh),
    "sin": (torch.sin, np.sin, sympy.sin),
    "abs": (torch.abs, np.abs, sympy.Abs),
}


def fit_affine_params(np_fn, x_np, y_np, n_restarts=10, seed=0):
    """
    Fit y ~= c * np_fn(a * x + b) + d by least squares, trying several
    random initializations (since np_fn may be non-monotonic, e.g. sin) and
    keeping the best fit by R^2.

    Returns:
        (a, b, c, d), r2 — or (None, -inf) if every restart failed to fit.
    """
    from scipy.optimize import curve_fit

    def model(x, a, b, c, d):
        return c * np_fn(a * x + b) + d

    y_mean = y_np.mean()
    ss_tot = np.sum((y_np - y_mean) ** 2) + 1e-12

    rng = np.random.default_rng(seed)
    best_params, best_r2 = None, -float("inf")
    for _ in range(n_restarts):
        p0 = [
            rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0]),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0]),
            rng.uniform(-1.0, 1.0),
        ]
        try:
            popt, _ = curve_fit(model, x_np, y_np, p0=p0, maxfev=10000)
        except Exception:
            continue
        y_pred = model(x_np, *popt)
        if not np.all(np.isfinite(y_pred)):
            continue
        ss_res = np.sum((y_np - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_params, best_r2 = tuple(popt), r2

    return best_params, best_r2
