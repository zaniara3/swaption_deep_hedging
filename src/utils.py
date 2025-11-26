import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from config import CONFIG


def compute_payoff(zc_prices, fixed_rate, dt, notional_value, swaptype='payer'):
    annuity = fixed_rate * dt * np.sum(zc_prices[:, 1:], axis=1)
    raw = zc_prices[:, 0] - zc_prices[:, -1]
    if swaptype == 'receiver':
        raw = -raw
        annuity = - annuity
    payoff = raw - annuity
    payoff[payoff < 0] = 0
    return notional_value * payoff


def penalty_function(he, method="mse", alpha=0.99, wghts=None, lambda_reg=1e-4, min_positive_ratio=0.1):
    if isinstance(he, np.ndarray):
        he = torch.from_numpy(he).float()

    # Core losses
    mse_loss = torch.mean(he ** 2)
    downside_loss = torch.mean(torch.clamp(he, min=0.0) ** 2)

    if method == "mse":
        loss = mse_loss

    elif method == "cvar":
        var_ = torch.quantile(he, alpha, interpolation="higher")
        excess = torch.clamp(he - var_, min=0.0)
        loss = var_ + excess.mean() / (1.0 - alpha)

    elif method == "downside":
        loss = downside_loss

    elif method == "root_downside":
        loss = (downside_loss + 1e-12).pow(0.5)

    elif method == "smooth_downside":
        smooth_loss = F.softplus(he, beta=10)
        loss = torch.mean(smooth_loss ** 2)

    elif method == "hybrid_downside":
        positive_ratio = torch.mean((he > 0).float()).item()
        if positive_ratio >= min_positive_ratio:
            loss = downside_loss
        else:
            lambda_mix = min(1.0, positive_ratio / (min_positive_ratio + 1e-8))
            loss = lambda_mix * downside_loss + (1 - lambda_mix) * mse_loss

    else:
        raise ValueError("Unknown penalty method")

    if wghts is not None:
        loss += lambda_reg * torch.sum(wghts ** 2)

    return loss


def prepare_inputs(cpath_train, factors, replicated_port_val, time2maturity):
    standardized_factor_path = StandardScaler().fit_transform(factors)
    standardized_replport_path = StandardScaler().fit_transform(replicated_port_val.reshape(-1, 1)).flatten()
    cpath_train[:, 0:3] = standardized_factor_path
    cpath_train[:, 3] = standardized_replport_path
    cpath_train[:, 4] = time2maturity
    return torch.tensor(cpath_train, dtype=torch.float32)


def compute_swap_pv_and_duration(zc_prices, fixed_rate, config=CONFIG):
    """
    Compute both present value and Macaulay duration of fixed leg of a swap.
    """
    dt = config['dt']
    notional = config['notional_value']
    num_steps = zc_prices.shape[1] - 1
    times = np.arange(1, num_steps + 1) * dt
    cashflows = fixed_rate * dt * np.ones(num_steps) * notional
    discounts = zc_prices[:, 1:]

    pv_cashflows = cashflows * discounts
    pv_total = np.sum(pv_cashflows, axis=1)

    weighted_times = times * pv_cashflows
    duration = np.zeros_like(pv_total)
    mask = pv_total != 0
    duration[mask] = np.sum(weighted_times[mask], axis=1) / pv_total[mask]

    return pv_total, duration


def compute_portfolio_pv_and_duration(zc_prices, swaps):
    portfolio_pv = np.zeros(zc_prices.shape[0])
    portfolio_duration = np.zeros(zc_prices.shape[0])
    total_weight = 0.0

    for swap in swaps:
        # Use swap-specific dt if provided, else fall back to global dt
        pv, duration = compute_swap_pv_and_duration(
            zc_prices[:, :swap['tenor'] + 1], swap['fixed_rate']
        )
        weight = swap['weight']
        portfolio_pv += weight * pv
        portfolio_duration += weight * pv * duration  # Weighted contribution to duration
        total_weight += weight

    # Normalize duration by total PV (if non-zero)
    mask = portfolio_pv != 0
    portfolio_duration[mask] /= portfolio_pv[mask]
    portfolio_duration[~mask] = 0  # Avoid NaN if PV=0

    return portfolio_pv, portfolio_duration


def evaluate_hedging_error(hedge_error):
    if isinstance(hedge_error, np.ndarray):
        he = torch.tensor(hedge_error, dtype=torch.float32)
    else:
        he = hedge_error.clone().detach().float()

    mean = torch.mean(he)
    std = torch.std(he)
    
    metrics = {
        "Mean": mean.item(),
        "Std": std.item(),
        "MSE": torch.mean(he ** 2).item(),
        "RMSE": torch.sqrt(torch.mean(he ** 2)).item(),
        "Downside (LPM2)": torch.mean(torch.clamp(he, min=0.0) ** 2).item(),
        "Root downside": torch.sqrt(torch.mean(torch.clamp(he, min=0.0) ** 2)).item(),
        "CVaR 95%": cvar(he, 0.95).item(),
        "CVaR 99%": cvar(he, 0.99).item(),
        "P(HE > 0)": torch.mean((he > 0).float()).item(),  # underhedging prob
        "Sharpe-like": (mean / std).item() if std > 0 else float('inf')
    }
    return metrics


def cvar(he, alpha=0.95):
    """
    Compute CVaR at level alpha (e.g., 0.95 or 0.99).
    Based on absolute hedge error.
    """
    var_alpha = torch.quantile(he, alpha, interpolation="higher")
    excess = torch.clamp(he - var_alpha, min=0.0)
    cvar_ = var_alpha + excess.mean() / (1.0 - alpha)
    return cvar_


def max_drawdown(path):
    peak = np.maximum.accumulate(path)
    drawdown = (peak - path) / peak
    return np.max(drawdown)


def compute_max_drawdown(portfolio_values):
    drawdowns = np.array([max_drawdown(path) for path in portfolio_values])
    average_drawdown = drawdowns.mean()
    return average_drawdown # Return as a scalar 


def solve_batch_linear_system(
    A: torch.Tensor,
    b: torch.Tensor,
    base_ridge: float = 1e-6,
    max_ridge: float = 1e-2,
    try_cholesky_first: bool = True,
) -> torch.Tensor:
    """
    Solve A x = b in batch, robust to singular/ill-conditioned A.

    Args:
        A: (..., M, M) batch of coefficient matrices.
        b: (..., M) or (..., M, K) right-hand side(s).
        base_ridge: minimal ridge added (scale-adaptive).
        max_ridge: cap for the adaptive ridge multiplier.
        try_cholesky_first: attempt fast SPD solve, else fall back to lstsq.

    Returns:
        x with shape matching b (i.e., (..., M) or (..., M, K)).
    """
    assert A.shape[-1] == A.shape[-2], "A must be square"
    M = A.shape[-1]
    b_was_vector = (b.ndim == A.ndim - 1)
    if b_was_vector:
        b = b.unsqueeze(-1)  # (..., M, 1)

    # Symmetrize to counter small asymmetries from numerical grads
    A = 0.5 * (A + A.transpose(-1, -2))

    # Adaptive ridge: scale with average diagonal magnitude per batch element
    # lam = clamp( base_ridge * mean(diag(A)), [base_ridge, max_ridge] )
    diag = torch.diagonal(A, dim1=-2, dim2=-1)
    mean_diag = diag.mean(dim=-1)  # (...,)
    lam = base_ridge * mean_diag.abs()
    lam = torch.clamp(lam, min=base_ridge, max=max_ridge)

    # Add ridge on the diagonal (broadcast over batches)
    I = torch.eye(M, dtype=A.dtype, device=A.device).expand_as(A)
    A_reg = A + lam[..., None, None] * I

    # First try a fast SPD path; if it fails, do least-squares
    if try_cholesky_first:
        try:
            L = torch.linalg.cholesky(A_reg)
            x = torch.cholesky_solve(b, L)
        except RuntimeError:
            # Not SPD — fall back to least-squares (handles rank-deficient)
            x = torch.linalg.lstsq(A_reg, b).solution
    else:
        x = torch.linalg.lstsq(A_reg, b).solution

    # Clean up numerical noise
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x.squeeze(-1) if b_was_vector else x


def project_boxed_l1_batch_torch(m, R, U, iters=40, eps=1e-12):
    """
    Euclidean projection (vectorized) of m (N,3) onto:
        0 <= x <= U    (box)
        sum(x, dim=1) <= R   (row-wise L1 ball)
    m, U: (N,3), R: (N,)
    """
    device, dtype = m.device, m.dtype
    zero = torch.zeros(1, device=device, dtype=dtype)

    m_clip = torch.minimum(torch.maximum(m, zero), U)     # (N,3)
    row_sum = m_clip.sum(dim=1)                           # (N,)
    feasible = row_sum <= (R + eps)

    x = torch.empty_like(m_clip)
    x[feasible] = m_clip[feasible]

    mask = ~feasible
    if mask.any():
        m_need = m_clip[mask]                             # (M,3)
        R_need = R[mask]                                  # (M,)
        U_need = U[mask]                                  # (M,3)

        lo = torch.zeros(m_need.shape[0], device=device, dtype=dtype)
        hi = m_need.max(dim=1).values

        for _ in range(iters):
            tau = 0.5 * (lo + hi)                         # (M,)
            z0  = torch.zeros_like(m_need)
            x_mid = torch.minimum(torch.maximum(m_need - tau[:, None], z0), U_need)
            s_mid = x_mid.sum(dim=1)
            gt = s_mid > R_need
            lo = torch.where(gt, tau, lo)
            hi = torch.where(gt, hi, tau)

        tau = hi
        z0  = torch.zeros_like(m_need)
        x_proj = torch.minimum(torch.maximum(m_need - tau[:, None], z0), U_need)
        x[mask] = x_proj

    return x


def apply_leverage_constraints(weights, prices, basis, per_swap_leverage, portfolio_limit, budget_cap, active_swaps, iters=40, eps=1e-12):
    """
    weights: (N,3) unconstrained notionals (can be +/-)
    prices:  (N,3) instrument prices (we use |.|)   [NO grad needed]
    basis:   (N,)  basis per path (|V_t| or |V_0|)  [NO grad needed]
    Returns constrained weights (N,3).
    """
    # We do NOT need gradients through the constraint mechanics:
    pabs  = torch.abs(prices).detach()
    basis = basis.detach()

    signs = torch.sign(weights)
    exp_uncon = torch.abs(weights) * pabs                      # (N,3)

    per_cap   = per_swap_leverage * (basis + budget_cap)       # (N,) basis + budget_cap                     # (N,)
    gross_cap = portfolio_limit   * (basis + budget_cap)                      # (N,)
    U = torch.stack([
        torch.where(active_swaps[0], per_cap, 0.0),
        torch.where(active_swaps[1], per_cap, 0.0)
    ], dim=1)

    exp_proj = project_boxed_l1_batch_torch(exp_uncon, gross_cap, U, iters=iters, eps=eps)
    weights_constrained = signs * exp_proj / (pabs + eps)
    if not active_swaps[0]: weights_constrained[:, 0] = 0.0
    if not active_swaps[1]: weights_constrained[:, 1] = 0.0
    return weights_constrained


def trading_intensity(weights: np.ndarray) -> float:
    """
    Compute the mean total L1 change in weights across time and assets, averaged over samples.

    Expected shape: (n_samples, n_times, n_assets)
    Equivalent to:
        np.mean(np.sum(np.sum(np.abs(np.diff(weights_OOS, axis=1)), axis=2), axis=1))

    Returns
    -------
    float
        Mean per-sample turnover (sum of absolute changes over time and assets).
    """
    w = np.asarray(weights[:, :, 0:-1])
    diffs = np.diff(w, axis=1)                  # changes over time
    per_sample_total = np.abs(diffs).sum(axis=(1, 2))
    return per_sample_total.mean()

