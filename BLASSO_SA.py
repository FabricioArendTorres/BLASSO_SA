#!/usr/bin/env python3
import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.linalg as linalg
import gig
import warnings


class BLASSO_SA:
    def __init__(self, X, Y, LAMBDA, BURNIN, NITER, NITER_cd, sigma2=1, T_0=1, T_n=0.01, thresh_low=0.2, gig_seed=1423):
        """Initialize the settings for the sampler.

        Arguments:
            X {ndarray} -- ndarray of the features, dim=(n,p)
            Y {ndarray} -- ndarray of the response, dim=(n)
            LAMBDA {float} -- sparsity parameter
            BURNIN {int} -- Number of iterations for burnin, must be >0
            NITER {int} -- Number of iterations for the Simulated Annealing, must be >0
            NITER_cd {int} -- Number of iterations for the cooled down system, must be >0

        Keyword Arguments:
            T_0 {int} -- Starting temperature (default: {1})
            T_n {float} -- Final temperature to be reached (default: {0.01})
            thresh_low {float} -- Defines the Credible Interval that is used for thresholding. Any Beta with a [thresh_low, 1-thresh_low] CI that covers 0 is estimated as 0. (default: {0.2})
            gig_seed {int} -- Seed for the Generalized Inverse Gaussian random sampler. (default: {1423})

        Raises:
            NotImplementedError -- [description]
        """
        assert(len(Y.shape) == 1)
        assert(X.shape[0] == Y.shape[0])
        assert(LAMBDA >= 0)
        assert(NITER > 0 and NITER_cd > 0 and T_n > 0 and T_0 >
               0 and thresh_low > 0 and thresh_low < 1)

        # TODO
        if(np.any(np.isnan(X)) or np.any(np.isnan(Y))):
            raise NotImplementedError("NaN not supported Yet.")

        if(np.abs(np.mean(Y)) > 0.0001):
            msg = "Are You sure Y is mean free?"
            warnings.warn(msg)

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
        self.T_list = np.zeros(NITER)

        # TODO proper initialization
        self.beta = np.ones(self.p)
        if(sigma2>0):
            self.sigma2=sigma2
        else:
            self.sigma2 = np.var(self.Y)

        self.did_run = False

    def _draw_beta(self, Y, X, X_gram, X_Y, T_inv, sigma2, T=1):
        A = (X_gram+np.diag(T_inv))
        # we really don't want to do this:
        # beta = sp.random.multivariate_normal(np.linalg.inv(A)@X_Y, T*sigma2*np.linalg.inv(A))

        L = np.linalg.cholesky(A)
        L_T = L.T

        tmp = linalg.solve_triangular(a=L, b=X_Y, lower=True)
        mu = linalg.solve_triangular(a=L_T, b=tmp)

        r = sp.random.standard_normal(size=self.p)
        # b ~ N(0, T*sigma^2*A^-1)
        b = linalg.solve_triangular(a=L_T, b=r*np.sqrt(T*sigma2))
        # Alternative: multiply with it afterwards..
        #  b = b * np.sqrt(T*sigma2)
        # beta ~ N(A^-1*X'y, T*sigma^2*A^-1)
        beta = mu+b

        return(beta)

    def _draw_sigma2(self, Y, X, beta, T_inv, n, p, T=1):

        tmp = Y-(X@beta)
        sigma2 = stats.invgamma.rvs(a=(0.5*(n-1+p)+1)/T - 1,
                                    scale=(0.5*(tmp.T@tmp) + beta.T@np.diag(T_inv)@beta)/T)
        return(sigma2)

    def _draw_T_inv(self, beta, sigma2, LAMBDA, T=1):
        """Draw the (Generalized) Inverse Gaussian distributed tau^-1

        Arguments:
            beta {float} -- [description]
            sigma2 {float} -- [description]
            LAMBDA {float} -- [description]

        Keyword Arguments:
            T {int} -- [description] (default: {1})
        """
        # parameters for inverse gaussian
        mus2 = np.abs(LAMBDA**2 * sigma2 / beta)
        lambdas = LAMBDA**2 * np.ones(self.p)

        # parameters for generalized inverse gaussian
        a = (lambdas/mus2)/T
        b = lambdas/T
        p_ = (-1.5/T + 1) * np.ones(self.p)

        T_inv = self.GIG.sample(psi=a, chi=b, lambda_=p_)
        return(T_inv)

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
        """Run the sampler!
        The estimated betas will be available from self.beta.
        """
        # init
        beta = self.beta
        sigma2 = self.sigma2
        X_gram = self.X.T@self.X
        X_Y = self.X.T @ self.Y

        # burnin
        for m in range(self.BURNIN):

            T_inv = self._draw_T_inv(beta, sigma2, self.LAMBDA)

            sigma2 = self._draw_sigma2(
                self.Y, self.X, beta, T_inv, self.n, self.p)

            beta = self._draw_beta(self.Y, self.X, X_gram, X_Y, T_inv, sigma2)

            # store results
            self.B_list[m, :] = beta
            self.sig2_list[m] = sigma2
            self.T_inv_list[m, :] = T_inv

        X_gram = self.X.T@self.X
        X_Y = self.X.T@self.Y

        sigma2 = np.mean(self.sig2_list[:self.BURNIN])

        # run sampler
        for m in range(self.NITER):
            # for the last NITER_cd iterations, leave the temperature fixed
            if(m < (self.NITER - self.NITER_cd)):
                T = BLASSO_SA.f_cool(m, self.T_0, self.a)

            T_inv = self._draw_T_inv(beta, sigma2, self.LAMBDA, T)

            # sigma2 = self._draw_sigma2(
            #     self.Y, self.X, beta, T_inv, self.n, self.p, T)

            beta = self._draw_beta(
                self.Y, self.X, X_gram, X_Y, T_inv, sigma2, T)

            # store results
            self.B_list[m+self.BURNIN, :] = beta
            self.sig2_list[m+self.BURNIN] = sigma2
            self.T_inv_list[m+self.BURNIN, :] = T_inv
            self.T_list[m] = T

        self.did_run = True
        self.beta = self._estimate_beta(self.thresh_low)

        return(self)

    @staticmethod
    def estimate_from_CI(var_hist, thresh_low):
        """Estimate the parameters from their history according to their Credible Interval. If the CI covers 0, the parameter is estimated as 0. Else, the median is taken.

        Arguments:
            var_hist {[type]} -- [description]
            thresh_low {[type]} -- [description]
        """
        assert(len(var_hist.shape) < 3)

        if(len(var_hist.shape) > 1):
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


def main():
    print("This script is not intended to be executed directly :).")


if __name__ == "__main__":
    main()
