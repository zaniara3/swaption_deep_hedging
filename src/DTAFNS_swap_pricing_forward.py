import numpy as np
from DTAFNS_zero_coupon_price_multipaths import DTAFNS_close


def swap_pricing_grad(param_kappa, param_theta, param_rho, param_sigma, param_lambda, p_factors_t_pricing,
                      t_pricing, param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value, stepsize=1 / 12,
                      swaptype='payer'):
    if t_pricing > param_T_alpha:
        raise ValueError('t_pricing can not be greater then T_alpha')
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))
    Np = p_factors_t_pricing.shape[0]
    DTAFNS_price = np.zeros((Np, len(times_swaps)))
    DTAFNS_derivative_x1 = np.zeros((Np, len(times_swaps)))
    DTAFNS_derivative_x2 = np.zeros((Np, len(times_swaps)))
    DTAFNS_derivative_x3 = np.zeros((Np, len(times_swaps)))
    swap_price = 0
    swap_x1, swap_x2, swap_x3 = (0, 0, 0)


    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=p_factors_t_pricing, kappa=param_kappa, theta=param_theta,
                              sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=t_pricing * stepsize,
                              delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        DTAFNS_derivative_x1[:, j] = dtafns.B1() * DTAFNS_price[:, j]
        DTAFNS_derivative_x2[:, j] = dtafns.B2() * DTAFNS_price[:, j]
        DTAFNS_derivative_x3[:, j] = dtafns.B3() * DTAFNS_price[:, j]


    if swaptype == 'receiver':
        swap_price = (p_notional_value *
                      (- DTAFNS_price[:, 0] + DTAFNS_price[:, -1]
                       + stepsize * param_fixed_rate * DTAFNS_price[:, 1:].sum(axis=1)))
        swap_x1 = -(p_notional_value * stepsize *
                    (- DTAFNS_derivative_x1[:, 0] + DTAFNS_derivative_x1[:, -1]
                     + stepsize * param_fixed_rate * DTAFNS_derivative_x1[:, 1:].sum(axis=1)))
        swap_x2 = -(p_notional_value * stepsize *
                    (- DTAFNS_derivative_x2[:, 0] + DTAFNS_derivative_x2[:, -1]
                     + stepsize * param_fixed_rate * DTAFNS_derivative_x2[:, 1:].sum(axis=1)))
        swap_x3 = -(p_notional_value * stepsize *
                    (- DTAFNS_derivative_x3[:, 0] + DTAFNS_derivative_x3[:, -1]
                     + stepsize * param_fixed_rate * DTAFNS_derivative_x3[:, 1:].sum(axis=1)))
    elif swaptype == 'payer':
        swap_price = (p_notional_value *
                      (DTAFNS_price[:, 0] - DTAFNS_price[:, -1]
                       - stepsize * param_fixed_rate * DTAFNS_price[:, 1:].sum(axis=1)))
        swap_x1 = -(p_notional_value * stepsize *
                    (DTAFNS_derivative_x1[:, 0] - DTAFNS_derivative_x1[:, -1]
                     - stepsize * param_fixed_rate * DTAFNS_derivative_x1[:, 1:].sum(axis=1)))
        swap_x2 = -(p_notional_value * stepsize *
                    (DTAFNS_derivative_x2[:, 0] - DTAFNS_derivative_x2[:, -1]
                     - stepsize * param_fixed_rate * DTAFNS_derivative_x2[:, 1:].sum(axis=1)))
        swap_x3 = -(p_notional_value * stepsize *
                    (DTAFNS_derivative_x3[:, 0] - DTAFNS_derivative_x3[:, -1]
                     - stepsize * param_fixed_rate * DTAFNS_derivative_x3[:, 1:].sum(axis=1)))

    swap_rate = (DTAFNS_price[:, 0] - DTAFNS_price[:, -1]) / (stepsize * DTAFNS_price[:, 1:].sum(axis=1))
        
    swap_Xs = np.vstack((swap_x1, swap_x2, swap_x3)).T

    return swap_price, swap_Xs, swap_rate

