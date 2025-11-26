import numpy as np


class DTAFNS_forward_measure:
    def __init__(self, Xt, kappa, theta, sigma, rho, T, t, delta_t, lam, seed=2025):
        self.Xt = Xt
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T
        self.t = t
        self.tau = (T - t) / delta_t
        self.Delta = delta_t
        self.lam = lam
        self.seed = seed

    def zeta0(self, r, tt):
        return (r - r ** tt) / (1 - r)

    def zeta1(self, r, tt):
        return (r - tt * (r ** tt) + (tt - 1) * r ** (tt + 1)) / (1 - r) ** 2

    def zeta2(self, r, tt):
        return (
                (-(tt - 1) ** 2 * r ** (tt + 2)
                 + (2 * tt ** 2 - 2 * tt - 1) * r ** (tt + 1)
                 - tt ** 2 * r ** tt + r ** 2 + r)
                / (1 - r) ** 3
        )

    def eta1(self):
        term1 = 0.5 * self.sigma[0, 0] * (self.tau - 1) * self.tau
        term2 = self.sigma[1, 1] * self.rho[0, 1] * (self.tau - 1 - self.zeta0(1 - self.lam, self.tau)) / self.lam
        z0 = self.zeta0(1 - self.lam, self.tau - 1)
        z1 = self.zeta1(1 - self.lam, self.tau - 1)
        term3 = self.sigma[2, 2] * self.rho[0, 2] * ((self.tau - 1 - (1 + z0)) / self.lam - z1)
        return self.Delta * self.sigma[0, 0] * (term1 + term2 + term3)

    def eta2(self):
        term1 = self.sigma[0, 0] * self.rho[1, 0] * self.zeta1(1 - self.lam, self.tau)
        z01 = self.zeta0(1 - self.lam, self.tau)
        z02 = self.zeta0((1 - self.lam) ** 2, self.tau)
        term2 = self.sigma[1, 1] * self.rho[1, 1] * (z01 - z02)
        z12 = self.zeta1((1 - self.lam) ** 2, self.tau - 1)
        term3 = self.sigma[2, 2] * self.rho[1, 2] * ((z01 - z02 / (1 - self.lam)) / self.lam - (1 + self.lam) * z12)
        return self.Delta * self.sigma[1, 1] * (term1 + term2 + term3)

    def eta3(self):
        term1 = self.sigma[0, 0] * self.rho[2, 0] * self.zeta1(1 - self.lam, self.tau)
        z01 = self.zeta0(1 - self.lam, self.tau)
        z02 = self.zeta0((1 - self.lam) ** 2, self.tau)
        term2 = self.sigma[1, 1] * self.rho[2, 1] * (z01 - z02)
        z12 = self.zeta1((1 - self.lam) ** 2, self.tau - 1)
        term3 = self.sigma[2, 2] * self.rho[2, 2] * ((z01 - z02 / (1 - self.lam)) / self.lam - (1 + self.lam) * z12)
        return self.Delta * self.sigma[2, 2] * (term1 + term2 + term3)

    def tau_eta3(self):
        term1 = self.sigma[0, 0] * self.rho[2, 0] * self.zeta2(1 - self.lam, self.tau)
        z11 = self.zeta1(1 - self.lam, self.tau)
        z12 = self.zeta1((1 - self.lam) ** 2, self.tau)
        term2 = self.sigma[1, 1] * self.rho[1, 0] * (z11 - z12) / self.lam
        z22 = self.zeta2((1 - self.lam) ** 2, self.tau)
        term3 = self.sigma[2, 2] * self.rho[2, 0] * ((z11 - z12) / self.lam - z22 / (1 - self.lam))
        return self.Delta * self.sigma[2, 2] * (term1 + term2 + term3) / (1 - self.lam)

    def M1(self):
        return self.Xt[0] - self.eta1()

    def M2(self):
        term1 = self.Xt[1] * (1 - self.lam) ** self.tau
        term2 = (self.theta[1] - self.theta[2]) * (1 - (1 - self.lam) ** self.tau) - self.eta2()
        term3 = self.lam * self.tau * self.Xt[2] * (1 - self.lam) ** (self.tau - 1)
        term4 = self.lam * self.theta[2] * (
                (1 - (1 - self.lam) ** self.tau) / self.lam - self.tau * (1 - self.lam) ** (self.tau - 1))
        term5 = - self.lam * self.tau_eta3()
        return term1 + term2 + term3 + term4 + term5

    def M3(self):
        lam1 = (1 - self.lam) ** self.tau
        return self.Xt[2] * lam1 + self.theta[2] * (1 - lam1) - self.eta3()

    def v11(self):
        return self.tau * self.sigma[0, 0] ** 2

    def v22(self):
        lam1 = 1 - self.lam
        lam2 = (1 - self.lam) ** 2
        term1 = self.sigma[1, 1] ** 2 * (1 + self.zeta0(lam2, self.tau))
        term2 = self.lam ** 2 * self.sigma[2, 2] ** 2 * self.zeta2(lam2, self.tau) / lam2
        term3 = 2 * self.sigma[1, 1] * self.lam * self.sigma[2, 2] * self.rho[1, 2] * self.zeta1(lam2, self.tau) / lam1
        return term1 + term2 + term3

    def v33(self):
        lam2 = (1 - self.lam) ** 2
        return self.sigma[2, 2] ** 2 * (1 + self.zeta0(lam2, self.tau))

    def v12(self):
        lam1 = 1 - self.lam
        term1 = self.sigma[0, 0] * self.sigma[1, 1] * self.rho[0, 1] * (1 + self.zeta0(lam1, self.tau))
        term2 = self.lam * self.sigma[0, 0] * self.sigma[2, 2] * self.rho[0, 2] * self.zeta1(lam1, self.tau) / lam1
        return term1 + term2

    def v13(self):
        lam1 = 1 - self.lam
        return self.sigma[0, 0] * self.sigma[2, 2] * self.rho[0, 2] * (1 + self.zeta0(lam1, self.tau))

    def v23(self):
        lam2 = (1 - self.lam) ** 2
        lam1 = 1 - self.lam
        term1 = self.sigma[1, 1] * self.sigma[2, 2] * self.rho[1, 2] * (1 + self.zeta0(lam2, self.tau))
        term2 = self.lam * self.sigma[2, 2] ** 2 * self.zeta1(lam2, self.tau) / lam1
        return term1 + term2

    def M_vec(self):
        return np.array([self.M1(), self.M2(), self.M3()])

    def V_mat(self):
        vmat = np.array([[self.v11(), self.v12(), self.v13()],
                         [self.v12(), self.v22(), self.v23()],
                         [self.v13(), self.v23(), self.v33()]])
        return vmat

    def generate_sample(self, sizes=10):
        np.random.seed(self.seed)
        M = self.M_vec()
        V = self.V_mat()
        if M.ndim == 1:
            samples = np.random.multivariate_normal(mean=M, cov=V, size=sizes)
        else:
            samples = np.stack([
                np.random.multivariate_normal(mean=M[:, i], cov=V[:, :, i], size=sizes)
                for i in range(M.shape[1])
            ], axis=1)

        return samples
