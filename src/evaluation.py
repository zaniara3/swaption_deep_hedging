import torch
import numpy as np
from config import CONFIG
from utils import penalty_function, prepare_inputs, apply_leverage_constraints


def evaluate_model(model, data, config):
    with torch.no_grad():
        # Extract data
        swap1 = torch.tensor(data["swap1_price"], dtype=torch.float32)
        swap2 = torch.tensor(data["swap2_price"], dtype=torch.float32)
        short = torch.tensor(data["short_rate"], dtype=torch.float32)
        factors = torch.tensor(data["factors"], dtype=torch.float32)
        payoff = torch.tensor(data["payoff"], dtype=torch.float32)
        swaption_price = torch.tensor(data["swaption_price"], dtype=torch.float32)
        swaps_expiry = data["swaps_expiries"]

        num_paths = swap1.shape[0]
        num_steps = config["T_alpha"]
        dt = config["dt"]

        # Initialize portfolio values
        port_val = torch.zeros((num_paths, num_steps + 1))
        port_val[:, 0] = swaption_price[:, 0]
        tc_multiplier = config["tc_multiplier"]

        input_buf = np.zeros((num_paths, config["input_size"]))
        asset_weights = torch.zeros((num_paths, num_steps, 3))  # w1, w2, w3, cash
        prev_weights = torch.zeros((num_paths, 2))
        transaction_cost_paths = torch.zeros((num_paths, num_steps))

        # Turnover tracking per path
        purchases_per_path = torch.zeros(num_paths)
        sales_per_path = torch.zeros(num_paths)
        portfolio_val_sum = torch.zeros(num_paths)

        for t in range(num_steps):
            active_cols = torch.tensor([t < swaps_expiry[0], t < swaps_expiry[1]], dtype=bool)
            # Prepare inputs for model
            ttm = (config["T_alpha"] - t) * dt
            inputs = prepare_inputs(input_buf, factors[:, t].numpy(), port_val[:, t].numpy(), ttm)

            # Predict weights: shape (num_paths, 3)
            prices_t = torch.stack([torch.abs(swap1[:, t]),
                                    torch.abs(swap2[:, t])], dim=1)  # (N,3)

            # Basis: dynamic = |V_t|, static = |V_0|
            if config.get("dynamic_basis", True):
                basis_t = torch.abs(port_val[:, t]) + 1e-8  # (N,)
            else:
                basis_t = torch.abs(port_val[:, 0]) + 1e-8
            weights = model(inputs)

            weights = apply_leverage_constraints(
            weights=weights,
            prices=prices_t,
            basis=basis_t,
            per_swap_leverage=config.get("per_swap_leverage", 1.5),
            portfolio_limit=config.get("portfolio_limit", 3.0),
            budget_cap=config.get("budget_cap", 1.0),
            active_swaps=active_cols,
            iters=40,
            eps=1e-12)

            # Transaction costs
            if t > 0:
                transaction_cost = (
                    torch.abs(weights[:, 0] - prev_weights[:, 0]) * torch.abs(swap1[:, t]) +
                    torch.abs(weights[:, 1] - prev_weights[:, 1]) * torch.abs(swap2[:, t])
                ) * tc_multiplier
            else:
                transaction_cost = torch.zeros(num_paths)

            # Update portfolio
            cash = (
                port_val[:, t] -
                weights[:, 0] * swap1[:, t] -
                weights[:, 1] * swap2[:, t] -
                transaction_cost
            )
            port_val[:, t + 1] = (
                weights[:, 0] * swap1[:, t + 1] +
                weights[:, 1] * swap2[:, t + 1] +
                cash * torch.exp(short[:, t] * dt)
            )

            # Save weights and costs
            asset_weights[:, t, 0:2] = weights
            asset_weights[:, t, 2] = cash
            transaction_cost_paths[:, t] = transaction_cost

            # Turnover per path
            if t > 0:
                delta_w1 = weights[:, 0] - prev_weights[:, 0]
                delta_w2 = weights[:, 1] - prev_weights[:, 1]

                purchases_per_path += (
                    torch.where(delta_w1 > 0, torch.abs(delta_w1) * torch.abs(swap1[:, t]), torch.tensor(0.0)) +
                    torch.where(delta_w2 > 0, torch.abs(delta_w2) * torch.abs(swap2[:, t]), torch.tensor(0.0))
                )

                sales_per_path += (
                    torch.where(delta_w1 < 0, torch.abs(delta_w1) * torch.abs(swap1[:, t]), torch.tensor(0.0)) +
                    torch.where(delta_w2 < 0, torch.abs(delta_w2) * torch.abs(swap2[:, t]), torch.tensor(0.0))
                )

            portfolio_val_sum += torch.abs(port_val[:, t])
            prev_weights = weights.clone()

        # Compute per-path turnover
        avg_portfolio_val_per_path = portfolio_val_sum / num_steps
        turnover_per_path = torch.minimum(purchases_per_path, sales_per_path) / (avg_portfolio_val_per_path + 1e-8)
        turnover_ratio = turnover_per_path.mean().item()

        # Final error
        error = payoff - port_val[:, -1]
        loss = penalty_function(error, config["penalty_method"])

        # Dynamic Tracking Error (DTE)
        squared_diff = (port_val[:, 1:] - swaption_price[:, 1:]) ** 2
        mse_per_path = squared_diff.mean(dim=1)
        dte_per_path = torch.sqrt(mse_per_path)
        dynamic_tracking_error = dte_per_path.mean().item()


    return (
        error.numpy(),
        loss.item(),
        asset_weights.numpy(),
        port_val.numpy(),
        transaction_cost_paths.numpy(),
        turnover_ratio,
        dynamic_tracking_error
    )


def project_boxed_l1_batch(m, R, U, iters=40, eps=1e-12):
    """
    Vectorized Euclidean projection of m (N,3) onto:
        0 <= x <= U   (box)
        sum(x, axis=1) <= R   (row-wise L1 ball)
    Returns x with same shape as m.
    """
    # Clip to box first
    m_clip = np.minimum(np.maximum(m, 0.0), U)           # (N, 3)
    row_sum = m_clip.sum(axis=1)                         # (N,)
    feasible = row_sum <= (R + eps)

    x = np.empty_like(m_clip)
    x[feasible] = m_clip[feasible]

    if np.any(~feasible):
        m_need = m_clip[~feasible]                       # (M, 3)
        R_need = R[~feasible]                            # (M,)
        U_need = U[~feasible]                            # (M, 3)

        lo = np.zeros(m_need.shape[0])
        hi = m_need.max(axis=1)

        # Bisection on tau for rows violating the L1 cap
        for _ in range(iters):
            tau = 0.5 * (lo + hi)                        # (M,)
            x_mid = np.minimum(np.maximum(m_need - tau[:, None], 0.0), U_need)
            s_mid = x_mid.sum(axis=1)
            gt = s_mid > R_need
            lo[gt] = tau[gt]
            hi[~gt] = tau[~gt]

        tau = hi
        x_proj = np.minimum(np.maximum(m_need - tau[:, None], 0.0), U_need)
        x[~feasible] = x_proj

    return x


def _solve_rect_least_squares_batched(A, b, base_ridge=1e-6, max_ridge=1e-2):
    """
    Batched rectangular least-squares with scale-adaptive ridge via augmentation:
        argmin_x ||A x - b||_2^2 + λ ||x||_2^2
    A: (N, R, C)   (R=3 here, C in {1,2,3})
    b: (N, R)
    returns x: (N, C)
    """
    assert A.ndim == 3 and b.ndim == 2 and A.shape[0] == b.shape[0] and A.shape[1] == b.shape[1]
    N, R, C = A.shape

    if C == 0:
        return torch.zeros((N, 0), dtype=A.dtype, device=A.device)

    # Scale-adaptive ridge based on column norms of A
    # (robust to tiny/degenerate gradients)
    col_norms = torch.linalg.norm(A, dim=1)              # (N, C)
    scale = (col_norms.mean(dim=1) + 1e-12)              # (N,)
    lam = torch.clamp(base_ridge * scale, min=base_ridge, max=max_ridge)  # (N,)

    # Augmented system: [A        ] x ≈ [b]
    #                   [sqrt(λ)I ]       [0]
    # Build per-batch augmented matrices
    sqrt_lam = torch.sqrt(lam).view(N, 1, 1)             # (N,1,1)
    I_C = torch.eye(C, dtype=A.dtype, device=A.device).expand(N, C, C)  # (N,C,C)
    A_aug_top = A                                         # (N,R,C)
    A_aug_bot = sqrt_lam * I_C                            # (N,C,C)
    A_aug = torch.cat([A_aug_top, A_aug_bot], dim=1)      # (N, R+C, C)

    b_aug_top = b                                         # (N,R)
    b_aug_bot = torch.zeros((N, C), dtype=b.dtype, device=b.device)
    b_aug = torch.cat([b_aug_top, b_aug_bot], dim=1)      # (N, R+C)

    # lstsq handles rank-deficiency per batch
    sol = torch.linalg.lstsq(A_aug, b_aug).solution       # (N, C)
    sol = torch.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)
    return sol


def _solve_rect_ls_with_smoothing(A, b, x_prev, base_ridge=1e-6, max_ridge=1e-2, mu=1e-2):
    N, R, C = A.shape
    col_norms = torch.linalg.norm(A, dim=1)          # (N,C)
    scale = (col_norms.mean(dim=1) + 1e-12)
    lam = torch.clamp(base_ridge * scale, min=base_ridge, max=max_ridge)

    I_C = torch.eye(C, dtype=A.dtype, device=A.device).expand(N, C, C)
    A_aug = torch.cat([
        A,
        torch.sqrt(lam).view(N,1,1)*I_C,
        torch.sqrt(torch.full_like(lam, mu)).view(N,1,1)*I_C
    ], dim=1)
    b_aug = torch.cat([
        b,
        torch.zeros((N, C), dtype=b.dtype, device=b.device),
        torch.sqrt(torch.full_like(lam, mu)).view(N,1)*x_prev
    ], dim=1)

    x = torch.linalg.lstsq(A_aug, b_aug).solution
    return torch.nan_to_num(x)


def evaluate_model_timeseries_rho_hedge_2x(
    data,
    config,
    swaps_expiry,              # iterable/list/array of length 3, in time steps
    on=(1, 2),
    unhedged=False,
    # Explicit leverage controls
    per_swap_leverage=CONFIG["per_swap_leverage"],    # per-swap dollar exposure cap = 1.5× basis
    portfolio_limit=CONFIG["portfolio_limit"],      # gross dollar exposure cap = 3.0× basis
    budget_cap=CONFIG["budget_cap"],                # per-swap dollar exposure cap = 1.5× basis
    dynamic_basis=True         # basis = |current portfolio value| if True, else initial option value
):
    """
    Dynamic hedging of a forward-start swaption using 3 forward-start swaps in a short-rate setup,
    with explicit leverage bounds enforced via a vectorized boxed-L1 projection and expiry-aware
    swap activation.

    swaps_expiry: (3,) array-like of ints; swap i is active iff t < swaps_expiry[i].
    Returns:
      loss.item(), error, asset_weights, repl_port_val, transaction_cost_paths, turnover_ratio, dynamic_tracking_error
    """
    # Validation
    if on[0] == on[1]:
        raise ValueError("The first and second elements of 'on' must not be equal.")
    if on[0] == 0 or on[1] == 0:
        raise ValueError("Indices are 1: X1, 2: X2, and 3: X3")
    # ---- Extract inputs
    swap1_price = data["swap1_price"]  # (num_paths, T_alpha+1)
    swap1_grad  = data["swap1_grad"]   # (num_paths, T_alpha, k=3)
    swap2_price = data["swap2_price"]
    swap2_grad  = data["swap2_grad"]
    short_rate  = data["short_rate"]   # (num_paths, T_alpha)
    swaption_price = data["swaption_price"]  # (num_paths, T_alpha+1) or (num_paths,)
    swaption_grad  = data["swaption_grad"]   # (num_paths, T_alpha, k>=4) uses idx 1,2,3 below
    payoff = data["payoff"]                  # (num_paths,)
    tc_multiplier = config.get("tc_multiplier", 0.0)

    # ---- Dimensions
    T_alpha = swap1_price.shape[1] - 1
    dt = 1 / 12
    num_paths = swap1_price.shape[0]
    device = torch.device("cpu")

    # ---- Storage
    repl_port_val = np.zeros((num_paths, T_alpha + 1))
    repl_port_val[:, 0] = swaption_price[:, 0] if swaption_price.ndim > 1 else swaption_price
    asset_weights = np.zeros((num_paths, T_alpha, 3))  # w1, w2, w3, cash
    prev_weights = np.zeros((num_paths, 2))
    transaction_cost_paths = np.zeros((num_paths, T_alpha))

    # Turnover per path
    purchases_per_path = np.zeros(num_paths)
    sales_per_path = np.zeros(num_paths)
    portfolio_val_sum = np.zeros(num_paths)

    eps = 1e-8
    swaps_expiry = np.asarray(swaps_expiry).reshape(2,)  # [e1, e2, e3], ints

    for t in range(T_alpha):
        # ---------- Active columns by expiry ----------
        # A swap is active at time t iff t < expiry_i
        active_cols = np.array([t < swaps_expiry[0], t < swaps_expiry[1]], dtype=bool)
        active_idx = np.where(active_cols)[0]
        m_active = active_idx.size

        # ---------- Build sensitivity matrix H (3 x 3) then select active columns (3 x m_active) ----------
        # Rows = risk-factor components; Cols = swaps
        g11 = swap1_grad[:, t, on[0] - 1]
        g12 = swap1_grad[:, t, on[1] - 1]
        g21 = swap2_grad[:, t, on[0] - 1]
        g22 = swap2_grad[:, t, on[1] - 1]

        H_full = np.stack(
            [
                np.stack([g11, g21], axis=1),
                np.stack([g12, g22], axis=1)
            ],
            axis=1
        )  # (num_paths, 3, 3)

        # Target gradient (swaption short-rate sensitivities)
        gr_np = np.stack([swaption_grad[:, t, on[0]], swaption_grad[:, t, on[1]]], axis=1)  # (num_paths, 3)

        # ---------- Solve for active weights (least-squares with ridge) ----------
        if unhedged or m_active == 0:
            weights = np.zeros((num_paths, 3))
        else:
            H_act = H_full[:, :, active_cols]                    # (num_paths, 3, m_active)
            # Torch solve (batched)
            H_act_t = torch.tensor(H_act, dtype=torch.float64, device=device)      # double for stability
            gr_t    = torch.tensor(gr_np, dtype=torch.float64, device=device)
            # w_act_t = _solve_rect_least_squares_batched(H_act_t, gr_t, base_ridge=1e-6, max_ridge=1e-2)  # (num_paths, m_active)
            w_act_t = _solve_rect_ls_with_smoothing(H_act_t, gr_t, prev_weights[:, active_idx], base_ridge=1e-6, max_ridge=1e-2)
            w_act   = w_act_t.detach().cpu().numpy()

            # Re-embed into 3-dim weight vector; inactive swaps receive 0
            weights = np.zeros((num_paths, 2))
            weights[:, active_cols] = w_act

        # ---------- Dollar exposures BEFORE constraints ----------
        p1_t = np.abs(swap1_price[:, t])
        p2_t = np.abs(swap2_price[:, t])

        w_abs = np.abs(weights)
        exp_uncon = np.stack(
            [w_abs[:, 0] * p1_t, w_abs[:, 1] * p2_t],
            axis=1
        )  # (num_paths, 3)
        signs = np.sign(weights)  # preserve long/short

        # ---------- Basis for leverage limits ----------
        if dynamic_basis:
            basis = np.maximum(np.abs(repl_port_val[:, t]), eps)  # per-path
        else:
            basis = np.maximum(np.abs(repl_port_val[:, 0]), eps)

        per_cap  = per_swap_leverage * (basis + budget_cap)      # (num_paths,)
        gross_cap = portfolio_limit * (basis + budget_cap)        # (num_paths,)

        # Per-component caps U_i (set to zero if swap is expired at t)
        U = np.stack([
            np.where(active_cols[0], per_cap, 0.0),
            np.where(active_cols[1], per_cap, 0.0)
        ], axis=1)  # (num_paths, 3)

        # ---------- Vectorized projection to boxed-L1 (respect expiry via U_i=0) ----------
        exp_proj = project_boxed_l1_batch(exp_uncon, gross_cap, U, iters=40, eps=1e-12)

        # Convert projected exposures back to weights (preserve signs)
        w1 = signs[:, 0] * (exp_proj[:, 0] / (p1_t + eps))
        w2 = signs[:, 1] * (exp_proj[:, 1] / (p2_t + eps))

        # Enforce expiry strictly: if inactive, force weight = 0
        if not active_cols[0]: w1 = np.zeros_like(w1)
        if not active_cols[1]: w2 = np.zeros_like(w2)

        weights = np.stack([w1, w2], axis=1)

        # ---------- Transaction costs ----------
        if t > 0:
            transaction_cost = (
                np.abs(weights[:, 0] - prev_weights[:, 0]) * p1_t +
                np.abs(weights[:, 1] - prev_weights[:, 1]) * p2_t
            ) * tc_multiplier
        else:
            transaction_cost = np.zeros(num_paths)

        # ---------- Portfolio update (cash accrual) ----------
        cash = (
            repl_port_val[:, t]
            - weights[:, 0] * swap1_price[:, t]
            - weights[:, 1] * swap2_price[:, t]
            - transaction_cost
        )

        repl_port_val[:, t + 1] = (
            weights[:, 0] * swap1_price[:, t + 1] +
            weights[:, 1] * swap2_price[:, t + 1] +
            cash * np.exp(short_rate[:, t] * dt)
        )

        # ---------- Save weights and costs ----------
        asset_weights[:, t, :2] = weights
        asset_weights[:, t, 2] = cash
        transaction_cost_paths[:, t] = transaction_cost

        # ---------- Turnover accounting ----------
        if t > 0:
            delta_w = weights - prev_weights
            purchases_per_path += (
                np.where(delta_w[:, 0] > 0, np.abs(delta_w[:, 0]) * p1_t, 0.0) +
                np.where(delta_w[:, 1] > 0, np.abs(delta_w[:, 1]) * p2_t, 0.0)
            )
            sales_per_path += (
                np.where(delta_w[:, 0] < 0, np.abs(delta_w[:, 0]) * p1_t, 0.0) +
                np.where(delta_w[:, 1] < 0, np.abs(delta_w[:, 1]) * p2_t, 0.0)
            )

        portfolio_val_sum += np.abs(repl_port_val[:, t])
        prev_weights = weights.copy()

    # ---- Per-path turnover
    avg_portfolio_val_per_path = portfolio_val_sum / max(T_alpha, 1)
    turnover_per_path = np.minimum(purchases_per_path, sales_per_path) / (avg_portfolio_val_per_path + eps)
    turnover_ratio = turnover_per_path.mean()

    # ---- Static error and loss
    error = payoff - repl_port_val[:, -1]
    loss = penalty_function(torch.tensor(error, dtype=torch.float32), config["penalty_method"])

    # ---- Dynamic Tracking Error vs. swaption price path
    squared_diff = (repl_port_val[:, 1:] - swaption_price[:, 1:]) ** 2
    mse_per_path = squared_diff.mean(axis=1)
    dte_per_path = np.sqrt(mse_per_path)
    dynamic_tracking_error = dte_per_path.mean()

    return (
        loss.item(),
        error,
        asset_weights,
        repl_port_val,
        transaction_cost_paths,
        turnover_ratio,
        dynamic_tracking_error
    )
