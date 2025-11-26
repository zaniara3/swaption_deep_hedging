import numpy as np
from DTAFNSModels import generate_path


def generate_paths_param(config, on='kappa', multiplier=1.05): # on ='theta'
    T_alpha = config["T_alpha"]
    seed = config["seed"]
    num_paths = config["num_paths"]

    kappa_p = np.array([[0.007484917, 0.0, 0.0], [0.0, 0.02878259, -0.02332395], [0, 0, 0.03536366]])
    theta_p = np.array([0, 0.03014341, 0.05050011])
    if on=='kappa':
        kappa_p = multiplier * np.array([[0.007484917, 0.0, 0.0], [0.0, 0.02878259, -0.02332395], [0, 0, 0.03536366]])
    elif on=='theta':
        theta_p = multiplier * np.array([0, 0.03014341, 0.05050011])

    rho = np.array([[1.0, -0.6303387, -0.4097114], [-0.6303387, 1.0, 0.2993069], [-0.4097114, 0.2993069, 1.0]])
    sigma_diag_vec = np.array([0.002680604, 0.004542753, 0.00701344])
    sigma = np.diag(sigma_diag_vec)
    initial_factors = np.array([-0.03123863, 0.03842199, 0.06882331])

    # factorpath_IS, short_rate_IS = generate_path(kappa_p, theta_p, rho, sigma, initial_factors, T_alpha, num_paths,
    #                                              theseed=seed)
    factorpath_OOS, short_rate_OOS = generate_path(kappa_p, theta_p, rho, sigma, initial_factors, T_alpha, num_paths,
                                                   theseed=seed + 100)

    return factorpath_OOS, short_rate_OOS
