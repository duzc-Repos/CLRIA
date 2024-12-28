import numpy as np
import ot
from scipy.optimize import fixed_point

class OTRegression(object):

    def __init__(self, pattern, regP=1, lambda_=2) -> None:
        """
        @parm pattern: list of matrix with equal size.
        @parm regP: int or float, regular parameters for Sinkhorm algorithm
        @parm lambda_: int or float, ridge regular parameters for $\beta$
        """
        self.X = pattern.copy()
        self.rank = len(self.X)
        self.regP = regP
        self.lambda_ = lambda_
        pass

    def _Entropy(self, P):
        tmp = P[P>0]
        return -(tmp * (np.log(tmp) - 1)).sum()
    def _squre_norm(self, beta):
        return (beta ** 2).sum()
    
    def get_M(self, beta):
        return np.exp( sum([ -beta[i] * self.X[i] for i in range(self.rank) ]) )

    def get_opt_obj_val(self, beta, P):
        M = self.get_M(beta)
        return (M * P).sum() - self.regP * self._Entropy(P) + self.lambda_/2 * self._squre_norm(beta)
    
    def get_opt_OT_dist(self, beta, P):
        M = self.get_M(beta)
        return (M * P).sum()
    
    def _check_src_dst(self, src, dst):
        src = np.array(src)
        dst = np.array(dst)
        assert np.allclose(src.sum(), dst.sum(), rtol=1e-8, atol=1e-8), f"sum(src) != sum(dst), {src.sum()}, {dst.sum()}"
        return src, dst

    def fit(self, src, dst, max_iter=(500, 1000, 500), delta=1e-16, min_iter=10, log=False):
        """
        @param src: array or list, source state or source active map
        @param dst: array or list, target state or target active map
        @param max_iter: tuple or list, 
        """

        def contractive_mapping(beta, lambda_, X, P):
            """
            @param beta: array
            @param lambda_: int or float
            #param X: list of matrix with equal size
            """
            rank = len(X)
            M = np.exp( sum([ -beta[i] * X[i] for i in range(rank) ]) )
            mid = P * M
            return np.array([ (X[i] * mid).sum() for i in range(rank) ]) / lambda_

        ## 00. Initialization
        src, dst = self._check_src_dst(src, dst)
        beta = np.ones(shape=self.rank) / self.rank

        ## 01. Iteration
        for k in range(max_iter[0]):
            
            ## Update P
            M = self.get_M(beta)
            P = ot.sinkhorn(a=src, b=dst, M=M, reg=self.regP, method="sinkhorn_log", numItermax=max_iter[1], log=False)
            #P = ot.emd(src, dst, M=M, numItermax=max_iter[1])

            ## Update beta
            beta = fixed_point(contractive_mapping, beta, args=(self.lambda_, self.X, P), maxiter=max_iter[2])

            ## Criterion
            if k == 0:
                dist_prev = self.get_opt_OT_dist(beta, P)
                continue
            
            dist_curr = self.get_opt_OT_dist(beta, P)
            rel_err = np.abs(dist_curr - dist_prev) / (dist_prev)
            if log:
                print(k, dist_curr, rel_err, sep="\t")
            if k < min_iter or rel_err > delta:
                dist_prev = dist_curr
            else:
                return beta, P
        
        msg = f"Failed to converge after {max_iter} iterations, value is {dist_curr}, rel_err is {rel_err}"
        raise RuntimeError(msg)

class UOTRegression(object):

    def __init__(self, pattern, regP=1, regm=1, lambda_=2) -> None:
        """
        @parm pattern: list of matrix with equal size.
        @parm regP: int or float, regular parameters for Sinkhorm algorithm
        @parm regm: int or float, regular parameters for mass constraints
        @parm lambda_: int or float, ridge regular parameters for $\beta$
        """
        self.X = pattern.copy()
        self.rank = len(self.X)
        self.regP = regP
        self.regm = regm
        self.lambda_ = lambda_
        pass

    def _Entropy(self, P):
        tmp = P[P>0]
        return -(tmp * (np.log(tmp) - 1)).sum()
    def _squre_norm(self, beta):
        return (beta ** 2).sum()
    def _KL_div(self, x, y):
        return (x * (np.log(x) - np.log(y) - 1)).sum() + y.sum()
    
    def get_M(self, beta):
        return np.exp( sum([ -beta[i] * self.X[i] for i in range(self.rank) ]) )
    def get_opt_UOT_dist(self, beta, P):
        M = self.get_M(beta)
        return (M * P).sum()
    def get_opt_obj_val(self, beta, P, src, dst):
        M = self.get_M(beta)
        kl_term = self.regm * self._KL_div(P.sum(axis=0), src) + self.regm * self._KL_div(P.sum(axis=1), dst)
        return (M * P).sum() - self.regP * self._Entropy(P) + kl_term + self.lambda_/2 * self._squre_norm(beta)
    
    def _check_src_dst(self, src, dst):
        src = np.array(src)
        dst = np.array(dst)
        return src, dst

    def fit(self, src, dst, beta_init=None, max_iter=(500, 1000, 500), delta=1e-16, min_iter=10, log=False):
        """
        @param src: array or list, source state or source active map
        @param dst: array or list, target state or target active map
        @beta_init: None, str or numpy.ndarray, initial beta value for iteration
        @param max_iter: tuple or list, 
        """

        def contractive_mapping(beta, lambda_, X, P):
            """
            @param beta: array
            @param lambda_: int or float
            #param X: list of matrix with equal size
            """
            rank = len(X)
            M = np.exp( sum([ -beta[i] * X[i] for i in range(rank) ]) )
            mid = P * M
            return np.array([ (X[i] * mid).sum() for i in range(rank) ]) / lambda_

        ## 00. Initialization
        src, dst = self._check_src_dst(src, dst)
        if beta_init is None:
            beta = np.ones(shape=self.rank) / self.rank
        elif beta_init == "random":
            beta = np.random.random(size=self.rank)
        else:
            beta = beta_init.copy()

        ## 01. Iteration
        for k in range(max_iter[0]):
            
            ## Update P
            M = self.get_M(beta)
            P = ot.sinkhorn_unbalanced(a=src, b=dst, M=M, reg=self.regP, reg_m=self.regm, 
                                       method="sinkhorn", numItermax=max_iter[1], log=False)
            #P = ot.emd(src, dst, M=M, numItermax=max_iter[1])

            ## Update beta
            beta = fixed_point(contractive_mapping, beta, args=(self.lambda_, self.X, P), maxiter=max_iter[2])

            ## Criterion
            if k == 0:
                dist_prev = self.get_opt_UOT_dist(beta, P)
                continue
            
            dist_curr = self.get_opt_UOT_dist(beta, P)
            rel_err = np.abs(dist_curr - dist_prev) / (dist_prev)
            if log:
                print(k, dist_curr, rel_err, sep="\t")
            if k < min_iter or rel_err > delta:
                dist_prev = dist_curr
            else:
                return beta, P
        
        msg = f"Failed to converge after {max_iter} iterations, value is {dist_curr}, rel_err is {rel_err}"
        raise RuntimeError(msg)
