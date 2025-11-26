import numpy as np
from DTAFNS_model_forward_measure import DTAFNS_forward_measure
from DTAFNS_zero_coupon_price_multipaths import DTAFNS_close


def swap_swaption_simulation(param_kappa_Q, param_theta_Q, param_rho, param_sigma, param_lambda, p_initial_factors,
                             param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value, stepsize=1 / 12,
                             npaths=100000,
                             swaptype='payer', seed=None):
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))
    dtafns_fm = DTAFNS_forward_measure(Xt=p_initial_factors, kappa=param_kappa_Q, theta=param_theta_Q,
                                       sigma=param_sigma,
                                       rho=param_rho, T=param_T_alpha * stepsize,
                                       t=0, delta_t=stepsize, lam=param_lambda, seed=seed)
    factorpath = dtafns_fm.generate_sample(sizes=npaths)

    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath, kappa=param_kappa_Q, theta=param_theta_Q, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa_Q, theta=param_theta_Q,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=0, delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))

    DTAFNS_price_0 = DTAFNS_price_sr[0]
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price


def swaprate_swaption_pricing(param_kappa, param_theta, param_rho, param_sigma, param_lambda, p_initial_factors,
                              factorpath_Talpha,
                              param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value, stepsize=1 / 12,
                              npaths=100000,
                              swaptype='payer', seed=None):
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))

    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath_Talpha, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=0, delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))

    DTAFNS_price_0 = DTAFNS_price_sr[0]
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price


def swaprate_swaption_pricing_for_grad_test(param_kappa, param_theta, param_rho, param_sigma, param_lambda,
                                            p_initial_factors, factorpath_Talpha,
                                            param_T_alpha, param_T_beta, param_fixed_rate, p_notional_value,
                                            stepsize=1 / 12,
                                            npaths=100000,
                                            swaptype='payer', seed=None, t_pricing=0):
    times_swaps = list(range(param_T_alpha, param_T_beta + 1))
    DTAFNS_price = np.zeros((npaths, len(times_swaps)))
    DTAFNS_price_sr = np.zeros(len(times_swaps))

    for j, TT in enumerate(times_swaps):
        dtafns = DTAFNS_close(Xt=factorpath_Talpha, kappa=param_kappa, theta=param_theta, sigma=param_sigma,
                              rho=param_rho, T=TT * stepsize, t=param_T_alpha * stepsize, delta_t=stepsize,
                              lam=param_lambda)
        DTAFNS_price[:, j] = dtafns.price_zero_coupon()
        dtafns_sr = DTAFNS_close(Xt=p_initial_factors.reshape(1, -1), kappa=param_kappa, theta=param_theta,
                                 sigma=param_sigma, rho=param_rho, T=TT * stepsize, t=t_pricing * stepsize,
                                 delta_t=stepsize,
                                 lam=param_lambda)
        DTAFNS_price_sr[j] = dtafns_sr.price_zero_coupon()

    swap_rate = (DTAFNS_price_sr[0] - DTAFNS_price_sr[-1]) / (stepsize * np.sum(DTAFNS_price_sr[1:]))
    DTAFNS_price_0 = DTAFNS_price_sr[0]
    payoff = 0
    if swaptype == 'payer':
        payoff = DTAFNS_price[:, 0] - DTAFNS_price[:, -1] - param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:],
                                                                                                 axis=1)
        payoff[payoff < 0] = 0
    elif swaptype == 'receiver':
        payoff = param_fixed_rate * stepsize * np.sum(DTAFNS_price[:, 1:], axis=1) - (
                DTAFNS_price[:, 0] - DTAFNS_price[:, -1])
        payoff[payoff < 0] = 0

    swaption_price = DTAFNS_price_0 * (p_notional_value * payoff).mean()
    return swaption_price, swap_rate, DTAFNS_price
