import numpy as np


def generate_path(kappa, theta, rho, sigma, initial_factors, nsteps, npaths, theseed=None):
    np.random.seed(theseed)

    # Check that rho is symmetric and positive-definite
    if not np.allclose(rho, rho.T):
        raise ValueError("rho must be symmetric.")
    try:
        cholrho = np.linalg.cholesky(rho)
    except np.linalg.LinAlgError:
        # Regularize rho if not positive-definite
        rho += np.eye(rho.shape) * 1e-12
        cholrho = np.linalg.cholesky(rho)

    # Precompute kappa @ theta for efficiency
    kappatheta = np.dot(kappa, theta)

    # Generate innovations
    innovarray = np.random.normal(size=(npaths, nsteps, 3))

    # Initialize output arrays
    xpath = np.zeros((npaths, nsteps + 1, 3))
    xpath[:, 0, :] = initial_factors

    for ss in range(nsteps):
        # Calculate next state while clipping for numerical stability
        drift = kappatheta - np.dot(kappa, xpath[:, ss, :].T).T
        noise = np.dot(sigma, np.dot(cholrho, innovarray[:, ss].T)).T

        # Clip noise to reduce risk of overflow (tune the bounds as needed)
        noise = np.clip(noise, -1e4, 1e4)
        drift = np.clip(drift, -1e4, 1e4)

        xpath[:, ss + 1, :] = np.clip(xpath[:, ss, :] + drift + noise, -1e6, 1e6)

    # Compute short rate, avoid nan/inf
    short_rate = np.nansum(xpath[:, :, :2], axis=2)
    short_rate = np.nan_to_num(short_rate, nan=0.0, posinf=1e6, neginf=-1e6)

    return xpath, short_rate
