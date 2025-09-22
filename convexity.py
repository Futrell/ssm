import torch
from typing import Iterable, List, Tuple

# ---------- utils: flatten/unflatten ----------
def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])

def _split_like(vec: torch.Tensor, like: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    outs, idx = [], 0
    for t in like:
        n = t.numel()
        outs.append(vec[idx:idx+n].view_as(t))
        idx += n
    return outs

# ---------- core: Hessian-vector product using autograd (Pearlmutter) ----------
def hvp(loss_closure, params: List[torch.Tensor], v_flat: torch.Tensor) -> torch.Tensor:
    """
    loss_closure(): returns a scalar loss computed at current 'params'.
    v_flat: flattened vector (same total size as params) to multiply by Hessian.
    """
    # First gradient (create graph to enable Hessian)
    loss = loss_closure()
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    # Dot with v, then second gradient
    dots = 0.0
    for g, v_chunk in zip(grads, _split_like(v_flat, params)):
        if g is None:
            # parameter not used in loss; treat as zeros
            continue
        dots = dots + (g.reshape(-1) * v_chunk.reshape(-1)).sum()
    Hv = torch.autograd.grad(dots, params, retain_graph=True, allow_unused=True)
    Hv = [torch.zeros_like(p) if (h is None) else h for h in Hv]
    return _flatten_tensors(Hv)

# ---------- dense Hessian assembly (exact, for small P) ----------
@torch.no_grad()
def dense_hessian(loss_closure, params: List[torch.Tensor]) -> torch.Tensor:
    P = sum(p.numel() for p in params)
    H = torch.zeros(P, P, device=params[0].device, dtype=params[0].dtype)
    eye = torch.eye(P, device=H.device, dtype=H.dtype)
    # We need grad graph inside hvp; so temporarily enable grad in loop by dropping no_grad
    torch._C._set_grad_enabled(True)
    try:
        for j in range(P):
            e_j = eye[:, j]
            Hj = hvp(loss_closure, params, e_j)  # column j
            H[:, j] = Hj
    finally:
        torch._C._set_grad_enabled(False)
    # Symmetrize to clean small numerical asymmetries
    H = 0.5 * (H + H.T)
    return H

# ---------- approximate extremes with (block) power iteration ----------
def _orthonormalize(Q: torch.Tensor, eps=1e-12) -> torch.Tensor:
    # Gram-Schmidt
    for i in range(Q.shape[1]):
        qi = Q[:, i:i+1]
        for j in range(i):
            qj = Q[:, j:j+1]
            qi = qi - (qj.T @ qi) * qj
        nrm = qi.norm()
        if nrm.item() < eps:
            # reinit a random vector if we hit near-zero
            qi = torch.randn_like(qi)
            nrm = qi.norm()
        Q[:, i:i+1] = qi / nrm
    return Q

def power_extremes(loss_closure, params: List[torch.Tensor], k: int = 5, iters: int = 50, seed: int = 0
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (largest_k, most_negative_k) eigenvalue estimates of the Hessian.
    Uses block power iteration on H and on (-H), with Hessian-vector products.
    """
    device = params[0].device
    dtype = params[0].dtype
    P = sum(p.numel() for p in params)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def block_power(sign: float) -> torch.Tensor:
        Q = torch.randn(P, k, device=device, dtype=dtype, generator=gen)
        Q = _orthonormalize(Q)
        torch._C._set_grad_enabled(True)
        try:
            for _ in range(iters):
                Z = []
                for i in range(k):
                    v = Q[:, i]
                    Hv = hvp(loss_closure, params, v)
                    Z.append(sign * Hv)  # sign=+1 for H, -1 for -H
                Z = torch.stack(Z, dim=1)  # (P, k)
                # Orthonormalize next basis
                Q = _orthonormalize(Z.clone())
        finally:
            torch._C._set_grad_enabled(False)
        # Rayleigh–Ritz on span(Q)
        # Build k×k projected matrix T = Q^T H Q (or Q^T(-H)Q if sign=-1)
        with torch.enable_grad():
            # One more set of H*Q to form T
            HQ = []
            for i in range(k):
                v = Q[:, i]
                Hv = hvp(loss_closure, params, v)
                HQ.append(Hv)
            HQ = torch.stack(HQ, dim=1)  # (P, k)
        T = Q.T @ (HQ)                      # (k, k) ~ Ritz matrix for H
        evals = torch.linalg.eigvalsh(sign * T)  # eigenvalues of H (sign=+1) or -H (sign=-1)
        return evals

    largest_k = block_power(sign=+1)        # largest k eigenvalues of H
    neg_of_min_k = block_power(sign=-1)     # largest k eigenvalues of -H  = - smallest k of H
    smallest_k = -neg_of_min_k.flip(0)      # sort ascending
    largest_k = largest_k.flip(0)           # sort descending
    return largest_k, smallest_k

def hessian_spectrum(loss_closure,
                     params: List[torch.Tensor],
                     approx_iters: int = 10,
                     seed: int = 0):
    P = sum(p.numel() for p in params)
    H = dense_hessian(loss_closure, params)
    evals = torch.linalg.eigvalsh(H)  # full spectrum (ascending)
    return evals.min().item(), evals.max().item()
    #return power_extremes(loss_closure, params, k=1, iters=approx_iters, seed=seed)
