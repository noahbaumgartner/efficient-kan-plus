import torch
import torch.nn.functional as F
import math

from .symbolic import SYMBOLIC_LIB, fit_affine_params


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation_class = base_activation
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # Which edges (out, in) have been replaced by a fitted symbolic
        # function (see fix_symbolic/auto_symbolic), and what that function is.
        self.register_buffer("symbolic_mask", torch.zeros(out_features, in_features))
        self.symbolic_funs = {}

        # Input to this layer's last forward call, cached for plot()/prune()/
        # auto_symbolic() so they can be called without re-supplying data.
        self._cached_input = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        self._cached_input = x.detach()

        if self.symbolic_funs:
            # Edges fixed to a symbolic function are excluded from the spline
            # matmul (masked out of the weights) and added back explicitly.
            numeric_mask = 1.0 - self.symbolic_mask
            base_output = F.linear(
                self.base_activation(x), self.base_weight * numeric_mask
            )
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                (self.scaled_spline_weight * numeric_mask.unsqueeze(-1)).view(
                    self.out_features, -1
                ),
            )
            output = base_output + spline_output

            idx_out = torch.tensor(
                [j for (j, _) in self.symbolic_funs.keys()],
                device=x.device,
                dtype=torch.long,
            )
            contributions = torch.stack(
                [
                    info["params"][2] * info["torch_fn"](info["params"][0] * x[:, i] + info["params"][1])
                    + info["params"][3]
                    for (_, i), info in self.symbolic_funs.items()
                ],
                dim=1,
            )
            output = output.index_add(1, idx_out, contributions)
        else:
            base_output = F.linear(self.base_activation(x), self.base_weight)
            spline_output = F.linear(
                self.b_splines(x).view(x.size(0), -1),
                self.scaled_spline_weight.view(self.out_features, -1),
            )
            output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def edge_outputs(self, x: torch.Tensor = None):
        """
        Per-edge contribution to the output, before summing over in_features.
        Differentiable (unlike most of the plot/prune/symbolic helpers) so it
        can also back an activation-based regularization loss; callers that
        only need it for analysis should wrap the call in `torch.no_grad()`.

        Args:
            x (torch.Tensor, optional): Input of shape (batch, in_features).
                Defaults to the input from the last forward() call.

        Returns:
            torch.Tensor: shape (batch, out_features, in_features).
        """
        if x is None:
            x = self._cached_input
        if x is None:
            raise RuntimeError(
                "No cached input available; call the model on data first, "
                "or pass x explicitly."
            )
        base = self.base_activation(x)  # (batch, in)
        base_out = base.unsqueeze(1) * self.base_weight.unsqueeze(0)  # (batch, out, in)
        splines = self.b_splines(x)  # (batch, in, coeff)
        spline_out = torch.einsum("bic,oic->boi", splines, self.scaled_spline_weight)
        edge_out = base_out + spline_out  # (batch, out, in)

        if self.symbolic_funs:
            override = torch.zeros_like(edge_out)
            mask = torch.zeros(
                edge_out.size(1), edge_out.size(2), dtype=torch.bool, device=edge_out.device
            )
            for (j, i), info in self.symbolic_funs.items():
                a, b, c, d = info["params"]
                override[:, j, i] = c * info["torch_fn"](a * x[:, i] + b) + d
                mask[j, i] = True
            edge_out = torch.where(mask.unsqueeze(0), override, edge_out)

        return edge_out

    def activation_regularization_loss(
        self, x: torch.Tensor = None, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ):
        """
        L1 + entropy sparsification loss computed from actual per-edge
        *activations* (mean |edge output| over the batch), matching the
        original KAN paper's regularizer rather than `regularization_loss`'s
        weight-magnitude proxy. `regularization_loss` avoids ever expanding to
        a (batch, out_features, in_features) tensor so the fast matmul-based
        forward stays untouched; this one deliberately pays for that
        expansion (via `edge_outputs`) because it tracks which edges the
        network actually uses given real data, not just which spline
        coefficients happen to be large — the latter can under- or
        over-penalize an edge relative to how much it actually varies over
        the input distribution. Prefer this when the weight-based loss isn't
        sparsifying as cleanly as the paper's examples suggest it should.
        """
        edge_out = self.edge_outputs(x)  # (batch, out, in)
        l1 = edge_out.abs().mean(dim=0)  # (out, in)
        reg_activation = l1.sum()
        p = l1 / reg_activation.clamp(min=1e-12)
        reg_entropy = -torch.sum(p * (p + 1e-12).log())
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy

    @torch.no_grad()
    def edge_importance(self, x: torch.Tensor = None):
        """Per-edge (out, in) importance score: std of that edge's output over the batch."""
        return self.edge_outputs(x).std(dim=0)

    @torch.no_grad()
    def edge_curve(self, in_idx: int, out_idx: int, x_values: torch.Tensor):
        """Evaluate a single edge's activation function phi_{out_idx,in_idx} at x_values (1D)."""
        if self.symbolic_mask[out_idx, in_idx] > 0:
            info = self.symbolic_funs[(out_idx, in_idx)]
            a, b, c, d = info["params"]
            return c * info["torch_fn"](a * x_values + b) + d

        x_full = torch.zeros(
            x_values.shape[0], self.in_features, device=x_values.device, dtype=x_values.dtype
        )
        x_full[:, in_idx] = x_values
        base = self.base_activation(x_values) * self.base_weight[out_idx, in_idx]
        splines = self.b_splines(x_full)[:, in_idx, :]  # (N, coeff)
        spline_out = splines @ self.scaled_spline_weight[out_idx, in_idx, :]
        return base + spline_out

    def _fit_edge_symbolic(self, in_idx: int, out_idx: int, fun_name: str, x: torch.Tensor = None):
        """Fit c * f(a*x+b) + d for candidate `fun_name` against this edge's current curve."""
        _, np_fn, _ = SYMBOLIC_LIB[fun_name]
        if x is None:
            x = self._cached_input
        x_edge = x[:, in_idx]
        with torch.no_grad():
            y_edge = self.edge_curve(in_idx, out_idx, x_edge)
        return fit_affine_params(
            np_fn, x_edge.detach().cpu().numpy(), y_edge.detach().cpu().numpy()
        )

    def fix_symbolic(
        self,
        in_idx: int,
        out_idx: int,
        fun_name: str,
        x: torch.Tensor = None,
        params: tuple = None,
        verbose: bool = True,
    ):
        """
        Replace edge (in_idx -> out_idx) with a symbolic function from
        SYMBOLIC_LIB, fitting c * f(a*x + b) + d to the edge's current
        learned curve (unless `params` is given explicitly).
        """
        torch_fn, _, sympy_fn = SYMBOLIC_LIB[fun_name]
        r2 = None
        if params is None:
            params, r2 = self._fit_edge_symbolic(in_idx, out_idx, fun_name, x=x)
            if params is None:
                raise RuntimeError(f"could not fit '{fun_name}' to edge ({in_idx}, {out_idx})")

        with torch.no_grad():
            self.symbolic_mask[out_idx, in_idx] = 1.0
        self.symbolic_funs[(out_idx, in_idx)] = dict(
            name=fun_name, torch_fn=torch_fn, sympy_fn=sympy_fn, params=tuple(params)
        )
        if verbose:
            msg = f"fixing ({in_idx},{out_idx}) with {fun_name}"
            if r2 is not None:
                msg += f", r2={r2}"
            print(msg)
        return r2

    def unfix_symbolic(self, in_idx: int, out_idx: int):
        with torch.no_grad():
            self.symbolic_mask[out_idx, in_idx] = 0.0
        self.symbolic_funs.pop((out_idx, in_idx), None)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    @torch.no_grad()
    def extend_grid(self, x: torch.Tensor, new_grid_size: int, margin=0.01):
        """
        Change the number of grid intervals (finer or coarser) while
        preserving the function currently represented by this layer.

        Unlike `update_grid`, which only repositions the existing grid
        points, this changes `grid_size` itself: the spline is evaluated on
        `x` under the old grid, a new grid with `new_grid_size` uniform
        intervals is built over the range of `x`, and the spline
        coefficients are refit (least squares) to match the old outputs on
        the new grid. This mirrors the "grid extension" technique from the
        original KAN paper, which allows training to start on a coarse grid
        and progressively refine it without losing the learned function.

        Args:
            x (torch.Tensor): Sample input of shape (batch_size, in_features)
                used to evaluate the current spline and fit the new one.
            new_grid_size (int): Number of grid intervals for the new grid.
            margin (float): Padding added to the range of `x` when building
                the new grid.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Evaluate the current (pre-extension) spline function on x.
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # Build a new uniform grid with `new_grid_size` intervals over the range of x.
        x_sorted = torch.sort(x, dim=0)[0]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / new_grid_size
        grid = (
            torch.arange(
                new_grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid_size = new_grid_size
        self.register_buffer("grid", grid.T.contiguous())
        self.spline_weight = torch.nn.Parameter(
            self.curve2coeff(x, unreduced_spline_output)
        )
        if self.enable_standalone_scale_spline:
            # The scaling has already been folded into unreduced_spline_output
            # (via scaled_spline_weight) and hence into the refit spline_weight,
            # so the scaler is reset to avoid applying it twice.
            self.spline_scaler = torch.nn.Parameter(torch.ones_like(self.spline_scaler))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    @classmethod
    def _from_layers(cls, layers):
        """Build a KAN directly from a list of (already configured) KANLinear layers."""
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.layers = torch.nn.ModuleList(layers)
        obj.grid_size = layers[0].grid_size
        obj.spline_order = layers[0].spline_order
        return obj

    @property
    def width(self):
        """Number of nodes in each layer, input to output: [in, hidden..., out]."""
        return [self.layers[0].in_features] + [layer.out_features for layer in self.layers]

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def activation_regularization_loss(self, x: torch.Tensor, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Sum of KANLinear.activation_regularization_loss() across all layers —
        the true, data-dependent sparsification penalty from the KAN paper,
        as opposed to regularization_loss()'s weight-based proxy. `x` is
        required and is propagated fresh through each layer via an extra
        forward pass (not read from the cached, detached inputs a prior
        forward() call left behind), so gradients reach every layer
        correctly; that extra pass is the efficiency this trades away.
        """
        total = 0.0
        cur = x
        for layer in self.layers:
            total = total + layer.activation_regularization_loss(cur, regularize_activation, regularize_entropy)
            cur = layer(cur)
        return total

    @torch.no_grad()
    def extend_grid(self, x: torch.Tensor, new_grid_size: int, margin=0.01):
        """
        Extend (or coarsen) every layer's grid to `new_grid_size` intervals,
        refitting each layer's spline coefficients to preserve the currently
        learned function. `x` is propagated layer by layer so that each
        layer is refit against the inputs it actually sees.
        """
        for layer in self.layers:
            layer.extend_grid(x, new_grid_size, margin=margin)
            x = layer(x)
        self.grid_size = new_grid_size

    def _layer_inputs(self, x: torch.Tensor = None):
        """Per-layer inputs: either propagated from `x`, or the cached inputs from the last forward()."""
        if x is not None:
            inputs = []
            cur = x
            for layer in self.layers:
                inputs.append(cur)
                cur = layer(cur)
            return inputs

        inputs = [layer._cached_input for layer in self.layers]
        if any(inp is None for inp in inputs):
            raise RuntimeError(
                "No cached activations found; call the model on data first "
                "(e.g. `model(x)`), or pass x explicitly."
            )
        return inputs

    @torch.no_grad()
    def _edge_relevance(self, x: torch.Tensor = None):
        """
        Per-layer edge importance (normalized to each layer's own max), plus
        per-node "relevance": whether this node's output actually reaches the
        final prediction, propagated backward from the output layer (which is
        defined as fully relevant). A node with a locally strong incoming
        edge but a locally weak outgoing edge — e.g. a hidden node whose only
        path to the output has been trained to near-zero — ends up with low
        relevance, and that low relevance is what should fade its incoming
        edges too, not just its outgoing one; otherwise the diagram can show
        a bold edge feeding a node whose contribution to y is negligible,
        which looks like a plotting bug rather than what it is (an edge that
        matters locally but leads nowhere).

        Returns:
            (layer_inputs, norm_importances, node_relevance) — norm_importances[l]
            is layer l's (out, in) importance in [0, 1]; node_relevance[l] is a
            (widths[l],) tensor in [0, 1] for the nodes between layer l-1 and l
            (node_relevance[0] is all 1s: inputs are never faded).
        """
        layer_inputs = self._layer_inputs(x)
        widths = self.width
        depth = len(self.layers)

        norm_importances = []
        for i, layer in enumerate(self.layers):
            imp = layer.edge_importance(layer_inputs[i])
            norm_importances.append(imp / imp.max().clamp(min=1e-12))

        node_relevance = [None] * (depth + 1)
        node_relevance[depth] = torch.ones(widths[depth])
        node_relevance[0] = torch.ones(widths[0])
        for l in range(depth - 1, 0, -1):
            # node_relevance[l+1] weights layer l's edges by whether their
            # destination (in layer l+1) matters; a node's own relevance is
            # then the best of its (now downstream-aware) outgoing edges.
            weighted = norm_importances[l] * node_relevance[l + 1].unsqueeze(1)  # (out, in)
            node_relevance[l] = weighted.max(dim=0).values  # (in,) = (widths[l],)

        return layer_inputs, norm_importances, node_relevance

    @torch.no_grad()
    def plot(
        self,
        x: torch.Tensor = None,
        in_vars=None,
        out_vars=None,
        title: str = None,
        num_pts: int = 100,
        beta: float = 3.0,
        ax=None,
    ):
        """
        Draw the KAN as a pykan-style diagram, bottom (input) to top (output):
        nodes are drawn as dots, each edge's learned 1D activation function is
        drawn as a bold curve inside a thin bordered box, and the boxes
        feeding into a node converge directly into that node's dot (every
        node here is a plain sum — this fork has no multiplication nodes —
        so unlike pykan there's no "+" marker to distinguish it). Edges are
        faded by `tanh(beta * edge_score)`, where `edge_score` is the edge's
        own local importance (std of its output over the batch, normalized to
        the layer's max) times its destination node's downstream relevance —
        recursively, whether that node's own outgoing edges eventually reach
        the output — so an edge that's locally "loud" but feeds a node whose
        only way out has trained to near-zero fades along with it, rather
        than showing up bold for a path that leads nowhere. Edges that barely
        matter fade to fully invisible, so a well-regularized, not-yet-pruned
        network already looks close to its pruned form. Edges fixed to a
        symbolic function (see
        fix_symbolic/auto_symbolic) are drawn in red instead of black,
        matching pykan's symbolic-vs-numeric color coding.

        Layout follows pykan's `MultKAN.plot`: within a layer, every node is
        spread evenly across a fixed-width row (so narrower layers are more
        spread out, wider layers denser), and every EDGE also gets its own
        evenly-spaced slot among that layer's `in_features * out_features`
        edges — independent of where its source/destination nodes sit. That
        slot assignment is what guarantees edge boxes never overlap, however
        wide a layer is: box size is derived from the single most crowded
        layer in the network and applied everywhere, so it's also visually
        consistent across layers.

        Args:
            x: representative input batch (batch, in_features). Defaults to
                the input from the last forward() call (so `model(x); model.plot()`
                works, matching pykan's usage).
            in_vars / out_vars: optional lists of label strings for the input
                / output nodes.
            beta: steepness of the importance -> alpha fade (pykan's default).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        import numpy as np

        layer_inputs, norm_importances, node_relevance = self._edge_relevance(x)
        widths = self.width
        depth = len(self.layers)

        total_width = 10.0  # every layer's nodes/edges are spread across this same span
        layer_gap = 3.0  # vertical gap between one node row and the next
        node_r = 0.15

        # Box size is fixed globally from the network's single most crowded
        # layer (most in_features * out_features edges), then capped so a
        # tiny network doesn't get comically large boxes.
        max_edges = max(widths[l] * widths[l + 1] for l in range(depth))
        box_w = min(0.85 * total_width / max_edges, 1.4)
        box_h = box_w  # square, like pykan's (equal x/y size + aspect="equal")

        def node_x(idx, width):
            return (idx + 0.5) / width * total_width - total_width / 2

        def layer_y(l):
            return l * layer_gap

        if ax is None:
            fig, ax = plt.subplots(figsize=(total_width * 0.85, layer_gap * depth * 0.85 + 1.5))
        else:
            fig = ax.figure

        node_pos = {}
        for l, w in enumerate(widths):
            y = layer_y(l)
            for i in range(w):
                node_pos[(l, i)] = (node_x(i, w), y)

        for l, layer in enumerate(self.layers):
            x_layer = layer_inputs[l]
            # An edge's displayed importance is its own local strength times
            # how relevant its destination node is downstream — an edge into
            # a node whose only way out has trained to near-zero fades along
            # with it, instead of showing up bold for a path that leads nowhere.
            edge_score = norm_importances[l] * node_relevance[l + 1].unsqueeze(1)  # (out, in)

            n_in, n_out = layer.in_features, layer.out_features
            n_edges = n_in * n_out
            y_src = layer_y(l)
            y_dst = layer_y(l + 1)
            y_box = (y_src + y_dst) / 2  # every node here is a plain sum, so no "+" marker is drawn

            def edge_x(i, j):
                slot = i * n_out + j
                return (slot + 0.5) / n_edges * total_width - total_width / 2

            for i in range(n_in):
                xi, _ = node_pos[(l, i)]
                col = x_layer[:, i]
                x_lo, x_hi = col.min().item(), col.max().item()
                if x_hi <= x_lo:
                    x_hi = x_lo + 1e-3
                xs = torch.linspace(x_lo, x_hi, num_pts, device=col.device)

                for j in range(n_out):
                    xj_node, _ = node_pos[(l + 1, j)]
                    alpha = float(np.tanh(beta * float(edge_score[j, i])))
                    ys = layer.edge_curve(i, j, xs).cpu().numpy()

                    # Edges fixed to a symbolic function are drawn in red,
                    # matching pykan's symbolic (red) vs numeric (black) split.
                    color = "red" if layer.symbolic_mask[j, i] > 0 else "black"

                    bx = edge_x(i, j)

                    ax.plot([xi, bx], [y_src, y_box - box_h / 2], color=color, linewidth=1.2, alpha=alpha, zorder=1)
                    ax.plot([bx, xj_node], [y_box + box_h / 2, y_dst - node_r], color=color, linewidth=1.2, alpha=alpha, zorder=1)

                    ax.add_patch(
                        Rectangle(
                            (bx - box_w / 2, y_box - box_h / 2), box_w, box_h,
                            facecolor="white", edgecolor="gray", linewidth=0.8, alpha=alpha, zorder=2,
                        )
                    )
                    span = ys.max() - ys.min()
                    span = span if span > 1e-9 else 1.0
                    ys_norm = (y_box - box_h / 2) + box_h * 0.12 + (ys - ys.min()) / span * box_h * 0.76
                    xs_norm = (bx - box_w / 2) + box_w * 0.1 + np.linspace(0, 1, num_pts) * box_w * 0.8
                    ax.plot(xs_norm, ys_norm, color=color, linewidth=2.2, alpha=alpha, zorder=3)

        for (l, i), (px, py) in node_pos.items():
            ax.add_patch(Circle((px, py), node_r, color="black", zorder=6))

        if in_vars is not None:
            for i, label in enumerate(in_vars):
                x0, y0_ = node_pos[(0, i)]
                ax.text(x0, y0_ - 0.4, str(label), ha="center", va="top", fontsize=11)
        if out_vars is not None:
            for j, label in enumerate(out_vars):
                x1, y1 = node_pos[(depth, j)]
                ax.text(x1, y1 + 0.4, str(label), ha="center", va="bottom", fontsize=11)

        ax.set_xlim(-total_width / 2 - 0.5, total_width / 2 + 0.5)
        ax.set_ylim(-0.6, layer_y(depth) + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)
        return fig, ax

    @torch.no_grad()
    def prune(self, x: torch.Tensor = None, node_th: float = 1e-2, verbose: bool = True):
        """
        Remove hidden-layer nodes whose incoming and outgoing edges are all
        unimportant (relative std below `node_th`), returning a new, smaller
        KAN. Input and output nodes are always kept.
        """
        layer_inputs = self._layer_inputs(x)
        importances = [layer.edge_importance(layer_inputs[i]) for i, layer in enumerate(self.layers)]
        normalized = [imp / imp.max().clamp(min=1e-12) for imp in importances]

        widths = self.width
        depth = len(self.layers)
        keep = [list(range(w)) for w in widths]

        for l in range(1, depth):
            incoming = normalized[l - 1].max(dim=1).values  # (widths[l],)
            outgoing = normalized[l].max(dim=0).values  # (widths[l],)
            node_score = torch.minimum(incoming, outgoing)
            kept = [i for i in range(widths[l]) if node_score[i].item() > node_th]
            if not kept:
                kept = [int(node_score.argmax())]
            keep[l] = kept
            if verbose:
                dropped = [i for i in range(widths[l]) if i not in kept]
                print(f"layer {l}: keeping {len(kept)}/{widths[l]} nodes (dropped {dropped})")

        new_layers = []
        for l, layer in enumerate(self.layers):
            in_idx = keep[l]
            out_idx = keep[l + 1]
            new_layer = KANLinear(
                in_features=len(in_idx),
                out_features=len(out_idx),
                grid_size=layer.grid_size,
                spline_order=layer.spline_order,
                scale_noise=layer.scale_noise,
                scale_base=layer.scale_base,
                scale_spline=layer.scale_spline,
                enable_standalone_scale_spline=layer.enable_standalone_scale_spline,
                base_activation=layer.base_activation_class,
                grid_eps=layer.grid_eps,
            )
            new_layer.grid.copy_(layer.grid[in_idx])
            new_layer.base_weight.data.copy_(layer.base_weight[out_idx][:, in_idx])
            new_layer.spline_weight.data.copy_(layer.spline_weight[out_idx][:, in_idx, :])
            if layer.enable_standalone_scale_spline:
                new_layer.spline_scaler.data.copy_(layer.spline_scaler[out_idx][:, in_idx])
            new_layers.append(new_layer)

        return KAN._from_layers(new_layers)

    def fix_symbolic(self, l: int, i: int, j: int, fun_name: str, x: torch.Tensor = None, verbose: bool = True):
        """Convenience wrapper: fix edge (i -> j) of layer `l` to a symbolic function. See KANLinear.fix_symbolic."""
        return self.layers[l].fix_symbolic(i, j, fun_name, x=x, verbose=verbose)

    def unfix_symbolic(self, l: int, i: int, j: int):
        self.layers[l].unfix_symbolic(i, j)

    def auto_symbolic(self, lib=None, r2_threshold: float = 0.9, verbose: bool = True):
        """
        For each edge in the network, try fitting every candidate function in
        `lib` (default: all of SYMBOLIC_LIB) as c * f(a*x + b) + d against the
        edge's current learned curve, and lock in the best one via
        fix_symbolic() if its R^2 clears `r2_threshold`.

        Requires the model to have been called on data first (so each layer
        has a cached input), or pass `x` explicitly.
        """
        if lib is None:
            lib = list(SYMBOLIC_LIB.keys())

        for l, layer in enumerate(self.layers):
            if layer._cached_input is None:
                raise RuntimeError(
                    "No cached activations found; call the model on data first "
                    "(e.g. `model(x)`) before auto_symbolic()."
                )
            x_layer = layer._cached_input
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    best = None
                    for name in lib:
                        params, r2 = layer._fit_edge_symbolic(i, j, name, x=x_layer)
                        if params is not None and (best is None or r2 > best[2]):
                            best = (name, params, r2)

                    if best is not None and best[2] >= r2_threshold:
                        name, params, r2 = best
                        layer.fix_symbolic(i, j, name, params=params, verbose=False)
                        if verbose:
                            print(f"fixing ({l},{i},{j}) with {name}, r2={r2}")
                    elif verbose:
                        name, _, r2 = best if best is not None else (None, None, float("-inf"))
                        print(
                            f"no symbolic fit for ({l},{i},{j}) cleared r2_threshold="
                            f"{r2_threshold} (best: {name}, r2={r2})"
                        )

    def symbolic_formula(self, var=None, simplify: bool = True, precision: int = 4):
        """
        Assemble a closed-form sympy expression per output, from edges that
        have been fixed via fix_symbolic()/auto_symbolic(). Edges still using
        the numeric spline are omitted from the formula (their contribution
        isn't representable in closed form).

        Args:
            precision: number of significant digits to round fitted
                constants to, so the printed formula is readable rather than
                showing raw floating-point noise.

        Returns:
            (expressions, var) — expressions[k] is the sympy formula for
            output k; var is the list of input symbol names used.
        """
        import sympy

        def r(v):
            return sympy.Float(v, precision)

        n_in = self.layers[0].in_features
        if var is None:
            var = [f"x{i + 1}" for i in range(n_in)]
        exprs = [sympy.Symbol(v) for v in var]

        for layer in self.layers:
            new_exprs = []
            for j in range(layer.out_features):
                terms = []
                for i in range(layer.in_features):
                    if layer.symbolic_mask[j, i] > 0:
                        info = layer.symbolic_funs[(j, i)]
                        a, b, c, d = info["params"]
                        terms.append(r(c) * info["sympy_fn"](r(a) * exprs[i] + r(b)) + r(d))
                new_exprs.append(sum(terms) if terms else sympy.Integer(0))
            exprs = new_exprs

        if simplify:
            exprs = [sympy.simplify(e) for e in exprs]
        return exprs, var
