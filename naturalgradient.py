import torch
from torch.optim import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# We use the torch.func API for stateless calls + JVP/VJP
try:
    from torch.func import functional_call, jvp, vjp
except Exception as e:
    raise RuntimeError(
        "This optimizer requires torch.func (PyTorch >= 2.0). "
        "Please upgrade your PyTorch."
    ) from e


def _params_to_pytree(model):
    # Returns an ordered dict (name->tensor) suitable for torch.func.functional_call
    return dict(model.named_parameters())

def _flat_to_pytree_like(flat, like_pytree):
    """Unflatten a flat vector into a pytree with same shapes as like_pytree."""
    out = {}
    idx = 0
    for k, p in like_pytree.items():
        numel = p.numel()
        out[k] = flat[idx:idx+numel].view_as(p)
        idx += numel
    return out

def _pytree_to_flat(pytree):
    return torch.cat([v.reshape(-1) for v in pytree.values()]) if pytree else torch.empty(0)


class NaturalGradientDescent(Optimizer):
    """
    Natural gradient optimizer that pulls back the exact Fisher on the intermediate automaton A.

    Args:
        params: iterable of model parameters (e.g., model.parameters())
        model: the module that owns those params
        lr: learning rate (natural step size)
        damping: Tikhonov damping λ (stabilizes/regularizes F_θ)
        cg_iters: max Conjugate Gradient iterations
        cg_tol: CG residual tolerance
        max_ls: (optional) backtracking line-search steps (0 disables)
        weight_decay: (optional) L2 on θ (added to loss gradient, not to Fisher)
    """
    def __init__(
        self,
        params,
        model,
        lr=0.1,
        damping=1e-4,
        cg_iters=50,
        cg_tol=1e-10,
        max_ls=0,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            damping=damping,
            cg_iters=cg_iters,
            cg_tol=cg_tol,
            max_ls=max_ls,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.model = model

    @torch.no_grad()
    def _gather_flat_grad(self):
        flats = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    flats.append(p.grad.reshape(-1))
                else:
                    flats.append(torch.zeros_like(p).reshape(-1))
        g = torch.cat(flats)
        # Add explicit L2 on parameters if requested (weight decay)
        total_decay = sum(group["weight_decay"] for group in self.param_groups)
        if total_decay > 0:
            flats = []
            for group in self.param_groups:
                wd = group["weight_decay"]
                for p in group["params"]:
                    flats.append((wd * p.detach()).reshape(-1))
            g = g + torch.cat(flats)
        return g

    def _flat_params(self):
        return torch.cat([p.detach().reshape(-1) for group in self.param_groups for p in group["params"]])

    def _flat_to_model(self, flat):
        # Write a flat vector back into live model parameters (in-place)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    def _make_param_pytree(self):
        return _params_to_pytree(self.model)

    def _Ftheta_matvec(self, params_pytree, automaton, v_flat, damping):
        """
        Compute y = (J^T F_A J + damping*I) v via:
          jv = J v
          wA = F_A jv = F_A J v
          y  = J^T wA + damping*v = J^T F_A J v + damping*v
        All without forming J or F_A explicitly.
        """
        # Convert v_flat into pytree tangent with same structure as params
        v_pytree = _flat_to_pytree_like(v_flat, params_pytree)

        # Define φ(params) = vec(A) in pytree form
        def phi(pytree_params):
            # Return vec(A) with grad
            with torch.enable_grad():
                automaton = type(self.model)(**pytree_params).fsa()
                return automaton.A.reshape(-1)

        # JVP: jv = J v
        jv, _ = jvp(phi, (params_pytree,), (v_pytree,))

        # Apply F_A matvec at A_value (treating Fisher as a function of current A)
        wA = automaton.apply_fisher(jv)  # same shape as A

        # VJP: y = J^T wA
        _, vjp_fn = vjp(phi, params_pytree)
        # vjp returns a tuple matching the input pytree; we have single pytree
        y_pytree = vjp_fn(wA.reshape(-1))[0]
        y_flat = _pytree_to_flat(y_pytree)

        return y_flat + damping * v_flat

    @torch.no_grad()
    def _cg_solve(self, matvec, b, iters, tol):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)
        for _ in range(iters):
            Ap = matvec(p)
            denom = torch.dot(p, Ap)
            if denom.abs() < 1e-30:
                break
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if rs_new.sqrt() < tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    @torch.no_grad()
    def step(self, closure=None):
        """
        Like LBFGS, this expects a `closure` to recompute loss and gradients.
        But it also works if you've already called loss.backward() before step().
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Gather gradient wrt θ (flattened)
        g = self._gather_flat_grad()
        
        # Snapshot current parameters as pytree
        params_pytree = self._make_param_pytree()
        # Compute current A once (no grad), to parameterize F_A
        with torch.no_grad():
            automaton = self.model.fsa()

        # Build matvec for (Fθ + λI)
        group = self.param_groups[0]
        damping = group["damping"]
        lr = group["lr"]
        cg_iters = group["cg_iters"]
        cg_tol = group["cg_tol"]
        max_ls = group["max_ls"]

        def matvec(v):
            return self._Ftheta_matvec(params_pytree, automaton, v, damping)

        # Solve (Fθ + λI) Δ = g
        delta = self._cg_solve(matvec, g, cg_iters, cg_tol)

        # Optional backtracking line search on the NG step (usually not needed).
        # We apply θ <- θ - lr * delta
        theta0 = self._flat_params()
        step = -lr * delta
        if closure is None or max_ls <= 0:
            self._flat_to_model(theta0 + step)
            return loss

        # Backtracking if a closure is provided
        # (Note: simple Armijo-like check; you can remove for simplicity.)
        f0 = loss.item()
        c = 1e-4
        t = 1.0
        for _ in range(max_ls):
            self._flat_to_model(theta0 + t * step)
            with torch.enable_grad():
                f_new = closure().item()
            if f_new <= f0 + c * t * torch.dot(g, step).item():
                break
            t *= 0.5
        # If we never accepted, we keep the last tried params anyway.
        return loss
