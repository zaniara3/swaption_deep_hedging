import numpy as np


class DTAFNS_close:
    def __init__(self, Xt, kappa, theta, sigma, rho, T, t, delta_t, lam):
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

    def B1(self):
        return self.tau

    def B2(self):
        return (1 - (1 - self.lam) ** self.tau) / self.lam

    def B3(self):
        return (1 - (1 - self.lam) ** (self.tau - 1)) / self.lam - (self.tau - 1) * (1 - self.lam) ** (self.tau - 1)

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

    def v11tau(self):
        return self.sigma[0, 0] ** 2 * self.tau * (self.tau - 1) * (2 * self.tau - 1) / 6

    def v22tau(self):
        return (
                self.sigma[1, 1] ** 2 / self.lam ** 2
                * (self.tau - 2 * self.B2() + (1 - (1 - self.lam) ** (2 * self.tau)) / (1 - (1 - self.lam) ** 2)))

    def v33tau(self):
        return (
                self.sigma[2, 2] ** 2 / self.lam ** 2
                * (
                        self.tau - 2
                        + self.zeta0((1 - self.lam) ** 2, self.tau - 1)
                        + self.lam ** 2 * self.zeta2((1 - self.lam) ** 2, self.tau - 1)
                        - 2 * self.zeta0(1 - self.lam, self.tau - 1)
                        - 2 * self.lam * self.zeta1(1 - self.lam, self.tau - 1)
                        + 2 * self.lam * self.zeta1((1 - self.lam) ** 2, self.tau - 1)
                )
        )

    def v12tau(self):
        return (
                self.rho[0, 1]
                * self.sigma[0, 0]
                * self.sigma[1, 1]
                / self.lam
                * (0.5 * self.tau * (self.tau - 1) - self.zeta1(1 - self.lam, self.tau))
        )

    def v13tau(self):
        return (
                self.rho[0, 2]
                * self.sigma[0, 0]
                * self.sigma[2, 2]
                / self.lam
                * (
                        0.5 * self.tau * (self.tau - 1)
                        - 1
                        - self.zeta0(1 - self.lam, self.tau - 1)
                        - (self.lam + 1) * self.zeta1(1 - self.lam, self.tau - 1)
                        - self.lam * self.zeta2(1 - self.lam, self.tau - 1)
                )
        )

    def v23taupart1(self):
        return (
                self.tau
                - 2
                - (2 - self.lam) * self.zeta0(1 - self.lam, self.tau - 1)
                + (1 - self.lam) * self.zeta0((1 - self.lam) ** 2, self.tau - 1)
        )

    def v23taupart2(self):
        return (
                -self.zeta1(1 - self.lam, self.tau - 1)
                + (1 - self.lam) * self.zeta1((1 - self.lam) ** 2, self.tau - 1)
        )

    def v23tau(self):
        return (
                self.rho[1, 2]
                * self.sigma[1, 1]
                * self.sigma[2, 2]
                * (self.v23taupart1() / self.lam ** 2 + self.v23taupart2() / self.lam)
        )

    def log_A(self):
        vtau = self.v11tau() + self.v22tau() + self.v33tau() + 2 * (self.v12tau() + self.v13tau() + self.v23tau())

        logA = -self.Delta * self.theta[1] * (self.B1() - self.B2()) + self.Delta * self.theta[
            2] * self.B3() + 0.5 * self.Delta ** 2 * vtau
        return logA

    def price_zero_coupon(self):
        return np.exp(self.log_A()) * np.exp(
            -self.Delta * (self.Xt[:, 0] * self.B1() + self.Xt[:, 1] * self.B2() + self.Xt[:, 2] * self.B3()))

    def yield_zero_coupon(self):
        return (
                -self.log_A() / (self.Delta * self.tau)
                + (self.Xt[:, 0] * self.B1() + self.Xt[:, 1] * self.B2() + self.Xt[:, 2] * self.B3()) / self.tau
        )
