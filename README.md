# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is mostly because it needs to expand all intermediate variables to perform the different activation functions.
For a layer with `in_features` input and `out_features` output, the original implementation needs to expand the input to a tensor with shape `(batch_size, out_features, in_features)` to perform the activation functions.
However, all activation functions are linear combination of a fixed set of basis functions which are B-splines; given that, we can reformulate the computation as activate the input with different basis functions and then combine them linearly.
This reformulation can significantly reduce the memory cost and make the computation a straightforward matrix multiplication, and works with both forward and backward pass naturally.

The problem is in the **sparsification** which is claimed to be critical to KAN's interpretability.
The authors proposed a L1 regularization defined on the input samples, which requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and is thus not compatible with the reformulation.
I instead replace the L1 regularization with a L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation.
The author's implementation indeed include this kind of regularization alongside the one described in the paper as well, so I think it might help.
More experiments are needed to verify this; but at least the original approach is infeasible if efficiency is wanted.

Those experiments turned out to matter: on the worked example below, the weight-based penalty leaves several extra edges visibly active even after training that the paper's own sparsification examples show as fully suppressed. This fork adds `activation_regularization_loss()` (see "Simplification techniques" below) as an opt-in alternative that pays for the `(batch, out, in)` expansion to compute the real, paper-matching penalty — the training-time cost the note above says is infeasible "if efficiency is wanted," offered here for when interpretability matters more than that.

Another difference is that, beside the learnable activation functions (B-splines), the original implementation also includes a learnable scale on each activation function.
I provided an option `enable_standalone_scale_spline` that defaults to `True` to include this feature; disable it will make the model more efficient, but potentially hurts results.
It needs more experiments.

2024-05-04 Update: @xiaol hinted that the constant initialization of `base_weight` parameters can be a problem on MNIST.
For now I've changed both the `base_weight` and `spline_scaler` matrices to be initialized with `kaiming_uniform_`, following `nn.Linear`'s initialization.
It seems to work much much better on MNIST (~20% to ~97%), but I'm not sure if it's a good idea in general.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. With uv installed, set up a local environment from the committed lockfile:

```bash
uv sync
```

This creates a `.venv` with all dependencies from `pyproject.toml`/`uv.lock` (PyTorch, torchvision, matplotlib, scipy, sympy, pytest, ipykernel). Run scripts and notebooks through it, e.g.:

```bash
uv run python examples/mnist.py
uv run jupyter notebook examples/grid_extension.ipynb
```

## Grid extension

`KANLinear`/`KAN` already support `update_grid`, which redistributes the existing grid points to better match the input distribution, but keeps `grid_size` fixed.

This fork adds `extend_grid(x, new_grid_size, margin=0.01)`, which implements the "grid extension" technique from the original KAN paper: it changes `grid_size` itself (to a finer or coarser grid) while refitting the spline coefficients (via least squares, like `curve2coeff`) so the function represented by the layer is preserved. This lets you start training on a coarse, cheap grid and later refine it to a finer grid without losing what was learned, e.g.:

```python
model = KAN([2, 4, 1], grid_size=5)
# ... train for a while ...
model.extend_grid(x_sample, new_grid_size=10)
# ... continue training with a finer grid ...
```

`x_sample` should be a representative batch of inputs to the model (`(batch_size, in_features)`); it's used both to evaluate the current spline function and to fit the new one, and is propagated layer by layer so every layer is refit against the inputs it actually sees.

## Simplification techniques: sparsification, visualization, pruning, and symbolification

The KAN paper (Section 2.5, "Simplifying KANs and Making them interactive") frames these as *simplification techniques*, driven interactively by a human rather than a black-box symbolic regressor. This fork adds pykan-style tooling for all four on top of the efficient forward pass:

- **`model.activation_regularization_loss(x)`** — a second sparsification loss, alongside the weight-based `regularization_loss()` from the section above. That one is free (no extra tensor, no extra forward pass) but, per the note above, doesn't concentrate importance onto as few edges as the paper's examples show, since spline-weight magnitude isn't the same thing as how much an edge's output actually varies over the data. This version computes real per-edge activations (deliberately paying for the `(batch, out, in)` tensor `regularization_loss()` avoids) and penalizes those instead, matching the paper's actual regularizer far more closely — compare the "before/after training" panels in `examples/simplification_techniques.ipynb` with each loss to see the difference directly.
- **`model.plot(x=None, in_vars=None, out_vars=None)`** draws the network as a diagram: one node per neuron, one small curve per edge showing that edge's learned 1D activation function, faded by how much it actually contributes to the output. Defaults to whatever input the model last saw (`model(x_train); model.plot()`), so you don't have to pass data in twice.
- **`model.prune(x=None, node_th=1e-2)`** removes hidden nodes whose incoming *and* outgoing edges are all below the importance threshold, returning a new, smaller `KAN` — predictions for the nodes that remain are unchanged. Input and output nodes are never pruned.
- **`model.fix_symbolic(l, i, j, name)`** / **`model.auto_symbolic(lib=None, r2_threshold=0.9)`** replace individual edges (or, for `auto_symbolic`, every edge that clears the R² threshold) with the best-fitting function from a small library (`x`, `x^2`, `x^3`, `x^4`, `exp`, `log`, `sqrt`, `tanh`, `sin`, `abs`), fit as `c * f(a*x + b) + d` via least squares (`scipy.optimize.curve_fit`).
- **`model.symbolic_formula(var=['x1', 'x2'])`** assembles the fixed edges into a closed-form `sympy` expression per output.

Under the hood, per-edge quantities (`KANLinear.edge_outputs`, `.edge_importance`, `.edge_curve`) are computed by briefly expanding to the `(batch, out_features, in_features)` tensor the rest of this codebase deliberately avoids — fine for analysis/visualization and for `activation_regularization_loss`, none of which are on the hot inference path. The normal `forward()` stays on the fast matmul path unless an edge has actually been fixed symbolically.

See [`examples/simplification_techniques.ipynb`](examples/simplification_techniques.ipynb) for the full workflow (train → visualize → prune → visualize → symbolify → recovered formula) on the same $f(x, y) = \exp(\sin(\pi x) + y^2)$ benchmark used above.
