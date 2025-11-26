from DTAFNS_swaption_pricing_forward import swaprate_swaption_pricing
from DTAFNS_swap_pricing_forward import swap_pricing_grad
import numpy as np
import torch


def price_swaption(factorpaths, config, is_oos=False):
    T_alpha = config["T_alpha"]
    T_beta = T_alpha + config["length_tenor"]
    dt = config["dt"]
    fixed_rate = config["fixed_rate"]
    notional_value = config["notional_value"]
    seed = config["seed"] + (100 if is_oos else 0)

    kappa_q = np.array([[0, 0, 0], [0, 0.02332395, -0.02332395], [0, 0, 0.02332395]])
    theta_q = np.array([0, 0.06326598, 0.07656804])
    rho = np.array([[1.0, -0.6303387, -0.4097114], [-0.6303387, 1.0, 0.2993069], [-0.4097114, 0.2993069, 1.0]])
    sigma = np.diag([0.002680604, 0.004542753, 0.00701344])
    lambda_param = 0.02332395
    initial_factors = np.array([-0.03123863, 0.03842199, 0.06882331])

    terminal_factors = factorpaths[:, T_alpha, :]

    price, swap_rate, zc_price = swaprate_swaption_pricing(
        kappa_q, theta_q, rho, sigma, lambda_param, initial_factors, terminal_factors,
        T_alpha, T_beta, fixed_rate, notional_value, dt, factorpaths.shape[0],
        config["swaptype"], seed
    )
    return price, swap_rate, zc_price


def compute_swap_rate(factorpaths, config, forward_start, tenor):
    T_alpha = forward_start
    dt = config["dt"]
    fixed_rate = 0.0
    notional_value = config["notional_value"]

    kappa_q = np.array([[0, 0, 0], [0, 0.02332395, -0.02332395], [0, 0, 0.02332395]])
    theta_q = np.array([0, 0.06326598, 0.07656804])
    rho = np.array([[1.0, -0.6303387, -0.4097114], [-0.6303387, 1.0, 0.2993069], [-0.4097114, 0.2993069, 1.0]])
    sigma = np.diag([0.002680604, 0.004542753, 0.00701344])
    lambda_param = 0.02332395

    T_beta = T_alpha + tenor
    num_paths = factorpaths.shape[0]

    _, _, swap_rate = swap_pricing_grad(
        kappa_q, theta_q, rho, sigma, lambda_param, factorpaths[:, 0, :],
        0, T_alpha, T_beta, fixed_rate, notional_value, dt, config["swaptype"])
    return swap_rate[0]


def compute_swap_gradients(factorpaths, config, fix_rate, forward_start, tenor):
    T_alpha = forward_start
    dt = config["dt"]
    fixed_rate = fix_rate
    notional_value = config["notional_value"]

    kappa_q = np.array([[0, 0, 0], [0, 0.02332395, -0.02332395], [0, 0, 0.02332395]])
    theta_q = np.array([0, 0.06326598, 0.07656804])
    rho = np.array([[1.0, -0.6303387, -0.4097114], [-0.6303387, 1.0, 0.2993069], [-0.4097114, 0.2993069, 1.0]])
    sigma = np.diag([0.002680604, 0.004542753, 0.00701344])
    lambda_param = 0.02332395

    T_beta = T_alpha + tenor
    num_paths = factorpaths.shape[0]

    length = min(config["T_alpha"], T_alpha) + 1

    price_matrix = np.zeros((num_paths, config["T_alpha"] + 1))
    grad_matrix = np.zeros((num_paths, config["T_alpha"] + 1, 3))

    for t in range(length):
        price_matrix[:, t], grad_matrix[:, t, :], _ = swap_pricing_grad(
            kappa_q, theta_q, rho, sigma, lambda_param, factorpaths[:, t, :],
            t, T_alpha, T_beta, fixed_rate, notional_value, dt, config["swaptype"]
        )
    return price_matrix, grad_matrix


def swaption_price_and_grads(factorpath, config, model, isgrad=False):
    dx = 0.01
    T_alpha = config["T_alpha"]
    dt = config["dt"]

    # tau_pricing for time-to-maturity feature
    tau_pricing = T_alpha * dt - np.arange(0, T_alpha + 1) * dt
    T_factors = np.insert(factorpath, 0, tau_pricing, axis=2)  # shape: (num_paths, T_alpha+1, num_features+1)

    num_paths = factorpath.shape[0]
    swaption_price_path = np.zeros((num_paths, T_alpha + 1))  # store prices
    grad = np.zeros((num_paths, T_alpha + 1, 4))  # store gradients

    # Compute swaption price for all times
    for i in range(T_alpha + 1):
        x_i = torch.from_numpy(T_factors[:, i, :].astype(np.float32))
        swaption_price_path[:, i] = model(x_i).squeeze(-1).detach().cpu().numpy()

    # Compute gradients if required
    if isgrad:
        for ind in range(4):  # compute derivative for each factor
            x_dx = T_factors.copy()
            x_dxb = T_factors.copy()
            x_dx[:, :, ind] += dx
            x_dxb[:, :, ind] -= dx

            for i in range(T_alpha + 1):
                x_dx_i = torch.from_numpy(x_dx[:, i, :].astype(np.float32))
                x_dxb_i = torch.from_numpy(x_dxb[:, i, :].astype(np.float32))
                f_dx = model(x_dx_i).squeeze(-1).detach().cpu().numpy()
                f_dxb = model(x_dxb_i).squeeze(-1).detach().cpu().numpy()
                grad[:, i, ind] = (f_dx - f_dxb) / (2 * dx)

    return swaption_price_path, grad



