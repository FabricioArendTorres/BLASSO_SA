#!/usr/bin/env pYthon3
import numpy as np
import scipy as sp
import scipy.stats.mstats as mstats
import scipy.linalg as linalg
import gig
import warnings


class BLASSO_SA:
    def __init__(self, X, Y, LAMBDA, BURNIN, NITER, NITER_cd, T_0=1, T_n=0.01, thresh_low=0.2, gig_seed=1423):
        assert(Y.shape[1] == 1)
        assert(X.shape[0] == Y.shape[0])
        assert(LAMBDA >= 0)
        assert(NITER > 0 and NITER_cd > 0 and T_n > 0 and T_0 >
               0 and thresh_low > 0 and thresh_low < 1)

        # TODO
        if(np.any(np.isnan(X)) or np.any(np.isnan(Y))):
            raise NotImplementedError("NaN not supported Yet.")

        if(np.abs(np.mean(Y)) > 0.0001):
            raise warnings.warn("Are You sure Y is mean free? mean(Y)={0}").format(
                np.round(np.mean(Y), 3))

        self.GIG = gig.GIG(gig_seed)

        self.n = X.shape[0]
        self.p = X.shape[1]

        self.LAMBDA = LAMBDA
        self.NITER = NITER
        self.NITER_cd = NITER_cd
        self.BURNIN = BURNIN
        self.thresh_low = thresh_low
        self.T_n = T_n
        self.T_0 = T_0
        self.X = X
        self.Y = Y

        self.a = np.power(T_n / T_0, 1/NITER)

        self.B_list = np.zeros((NITER + BURNIN, self.p))
        self.sig2_list = np.zeros((NITER + BURNIN))
        self.T_inv_list = np.zeros((NITER + BURNIN, self.p))

        # TODO proper initialization
        self.beta = np.ones(self.p)
        self.sigma2 = np.var(self.Y)

        self.did_run = False

    def _draw_beta(self, Y, X, X_gram, X_Y, T_inv, sigma2, T=1):
        A = (X_gram+T_inv)
        L = np.linalg.cholesky(A)
        L_T = L.T

        tmp = linalg.solve_triangular(a=L, b=X_Y, lower=True)
        mu = linalg.solve_triangular(a=L_T, b=tmp)

        r = sp.random.standard_normal(size=self.p)
        # b ~ N(0, T*sigma^2*A^-1)
        b = linalg.solve_triangular(a=L_T, b=r/np.sqrt(T*sigma2))
        # beta ~ N(A^-1*X'y, T*sigma^2*A^-1)
        beta = mu+b
        return(beta)

    def _draw_sigma(self, Y, X, beta, T_inv, n, p, T=1):
        raise NotImplementedError()

    def _draw_T_inv(self, beta, sigma2, LAMBDA, T=1):
        raise NotImplementedError()

    @staticmethod
    def f_cool(m, T0, a):
        """Using exponential cooling schedule T_m = T_0 * a^m

        Arguments:
            m {int} -- index of T_m
            T0 {float} -- Starting Temperature
            a {float} -- Factor of the cooling speed, 0<a<=1

        Returns:
            float -- Temperature at time point m
        """
        return T0 * pow(a, m)

    def run(self):
        # init
        beta = self.beta
        sigma2 = self.sigma2

        # burnin
        for m in range(self.BURNIN):

            T_inv = self._draw_T_inv(beta, sigma2, self.LAMBDA)

            sigma2 = self._draw_sigma(
                self.Y, self.X, beta, T_inv, self.n, self.p)

            beta = self._draw_beta(self.Y, self.X, T_inv, sigma2)

            # store results
            self.B_list[m, :] = beta
            self.sig2_list[m] = sigma2
            self.T_inv_list[m, :] = T_inv

        X_gram = self.X.T@self.X
        X_Y = X.T@self.Y

        # run sampler
        for m in range(self.NITER):
            # for the last NITER_cd iterations, leave the temperature fixed
            if(m < (self.NITER - self.NITER_cd)):
                T = BLASSO_SA.f_cool(m, self.T_0, self.a)

            T_inv = self._draw_T_inv(beta, sigma2, self.LAMBDA, T)

            sigma2 = self._draw_sigma(
                self.Y, self.X, beta, T_inv, self.n, self.p, T)

            beta = self._draw_beta(
                self.Y, self.X, X_gram, X_Y, T_inv, sigma2, T)

            # store results
            self.B_list[m, :] = beta
            self.sig2_list[m] = sigma2
            self.T_inv_list[m, :] = T_inv

        self.did_run = True
        self.beta = self._estimate_beta(self.thresh_low)

        return(self)

    @staticmethod
    def estimate_from_CI(var_hist, thresh_low):
        assert(len(var_hist.shape < 3))

        if(len(var_hist.shape > 1)):
            p = var_hist.shape[1]
        else:
            p = 1
        qua = [thresh_low, 0.5, 1-thresh_low]

        CI = mstats.mquantiles(a=var_hist, prob=qua, axis=0, alphap=1, betap=1)

        # unsignificant variables (according to our CI) are zero
        CI_sign = (CI[0, :] * CI[2, :]) > 0
        var_est = np.zeros(p)
        # estimate significant variables with median
        w = np.where(CI_sign)
        var_est[w] = CI[1, w]

        return(var_est)

    def _estimate_beta(self, thresh_low):
        beta_est = BLASSO_SA.estimate_from_CI(
            self.B_list[-self.NITER_cd:, ], thresh_low)
        return(beta_est)

    def predict(self, X):
        assert(self.did_run)
        assert(X.shape[1] == self.p)

        y_pred = X@self.beta
        return(y_pred)


def main():
    print("In Progress :)")


if __name__ == "__main__":
    main()
