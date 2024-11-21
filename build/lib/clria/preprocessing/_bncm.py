import numpy as np
import scipy as sp


class BNCM(object):
    def __init__(self):
        pass

    def proximal_to_distance(self, SC):
        """
        ref: Inferring neural signalling directionality from undirected structural connectomes
        """
        return -np.log( SC / (SC.max()+SC[SC>0].min()) )
    
    def calc_navigation(self, L, D, max_hops):
        """
        https://github.com/brainlife/BCT/blob/main/BCT/2019_03_03_BCT/navigation_wu.m
        """
        N = L.shape[0]
        paths = [[ [] for i in range(N) ] for j in range(N)]
        PL_bin = np.zeros(shape=(N, N))
        PL_wei = np.zeros(shape=(N, N))
        PL_dis = np.zeros(shape=(N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    curr_node = i
                    last_node = curr_node
                    target = j
                    paths[i][j].append(curr_node)

                    pl_bin, pl_wei, pl_dis = 0, 0, 0
                    while curr_node != target:
                        neighbours_idx = np.where(~np.isnan(L[curr_node, :]))[0]
                        if len(neighbours_idx) != 0:
                            min_index = np.argmin(D[target, neighbours_idx])
                            next_node = neighbours_idx[min_index]

                        if len(neighbours_idx) == 0 or next_node == last_node or pl_bin > max_hops:
                            pl_bin = np.Inf
                            pl_wei = np.Inf
                            pl_dis = np.Inf
                            break

                        paths[i][j].append(next_node)
                        pl_bin += 1
                        pl_wei += L[curr_node, next_node]
                        pl_dis += D[curr_node, next_node]

                        last_node = curr_node
                        curr_node = next_node

                    PL_bin[i, j] = pl_bin
                    PL_wei[i, j] = pl_wei
                    PL_dis[i, j] = pl_dis
        return PL_wei
    
    def calc_communicability(W):
        degree_sqrt = np.sqrt(W.sum(axis=1, keepdims=True))
        W_norm = W / (degree_sqrt.dot(degree_sqrt.T))
        return sp.linalg.expm(W_norm)

    def calc_TransCost(self, D, G, lambda_):
        """
        ref: https://github.com/aiavenak/lambda_spectrum
        """
        D = D.copy()
        G = G.copy()

        D[np.isinf(D)] = 0
        D[np.isnan(D)] = 0
        N = D.shape[1]

        Ctrans = np.zeros(shape=(N, N))
        for t in range(N):
            Gt = G[:, [t]].T
            Gt = Gt + D
            
            FG = np.exp(-(lambda_*Gt + D)) * (D>0)
            #FG = 1 / (lambda_*Gt + D) * (D>0)
            z_lamb_t = FG.sum(axis=1, keepdims=True)
            P = FG / z_lamb_t

            idx = np.arange(N) != t
            Q = P[idx, :][:, idx].copy()
            FM = np.linalg.inv(np.eye(N-1) - Q)
            Ctrans[idx, t] = FM.dot( (P[idx, :] * D[idx, :]).sum(axis=1) )

        return Ctrans









