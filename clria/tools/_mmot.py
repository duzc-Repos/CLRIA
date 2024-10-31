import numpy as np
import tensorly as tl
import os

class OBJFUNC(object):
    def __init__(self, L, R, M, TL, TR, epsilon=0, tau1=1, tau2=1, delta=1e-50, eps=1e-100,
                 device="numpy", verbose=False, log=False):

        device = self.get_device(device)
        tl.set_backend(device)
        L = tl.tensor(L, device="cuda", dtype=tl.backend.float64)
        L = tl.where(L>eps, L, eps)
        R = tl.tensor(R, device="cuda", dtype=tl.backend.float64)
        R = tl.where(R>eps, R, eps)
        if isinstance(M, tuple):
            M_type = tuple
            M1 = tl.tensor(M[0], device="cuda", dtype=tl.backend.float64)
            M2 = tl.tensor(M[1], device="cuda", dtype=tl.backend.float64)
            M = (M1, M2)
        else:
            M_type = "array"
            M = tl.tensor(M, device="cuda", dtype=tl.backend.float64)
        TL = tl.tensor(TL, device="cuda", dtype=tl.backend.float64)
        TR = tl.tensor(TR, device="cuda", dtype=tl.backend.float64)

        self.L = L
        self.R = R
        self.M = M
        self.M_type = M_type
        self.TL = TL
        self.TR = TR
        self.epsilon = epsilon
        self.tau1 = tau1
        self.tau2 = tau2
        self.delta = delta

        ##
        self.device = device
        self.verbose = verbose
        self.log = log

        ##
        self.flag_epsilon = epsilon == 0
        self.TLsum1 = TL.sum(axis=1, keepdims=True)
        self.TRsum1 = TR.sum(axis=1, keepdims=True)
        self.const_T = self.tau1 * self.TLsum1 + self.tau2 * self.TRsum1
        self.logL = tl.log(L)
        self.logR = tl.log(R)
        self.Lsum = tau1 * self.L.sum()
        self.Rsum = tau2 * self.R.sum()
        pass

    def f(self, A, B, C):
        Csum0 = C.sum(axis=0, keepdims=True)
        #print(self.M.shape, A.shape, B.shape, C.shape)
        if self.M_type == "array":
            return (self.M * tl.matmul(A*Csum0, B.T)).sum()
        else:
            return ( tl.matmul(self.M[1].T, B) * tl.matmul(self.M[0].T, A*Csum0) ).sum()
    def grad_f(self, A, B, C):
        Csum0 = C.sum(axis=0, keepdims=True)
        if self.M_type == "array":
            grad_A = tl.matmul(self.M, B*Csum0)
            grad_B = tl.matmul(self.M.T, A*Csum0)
            grad_C = (A*tl.matmul(self.M, B)).sum(axis=0, keepdims=True)
        else:
            grad_A = tl.matmul(self.M[0], tl.matmul(self.M[1].T, B*Csum0))
            grad_B = tl.matmul(self.M[1], tl.matmul(self.M[0].T, A*Csum0))
            grad_C = (tl.matmul(self.M[0].T, A) * tl.matmul(self.M[1].T, B)).sum(axis=0, keepdims=True)
        #if self.verbose:
        #    print(grad_A.shape, grad_B.shape, grad_C.shape)
        return [grad_A, grad_B, grad_C]
    def grad_fA(self, A, B, C):
        Csum0 = C.sum(axis=0, keepdims=True)
        if self.M_type == "array":
            return tl.matmul(self.M, B*Csum0)
        else:
            return tl.matmul(self.M[0], tl.matmul(self.M[1].T, B*Csum0))
    def grad_fB(self, A, B, C):
        Csum0 = C.sum(axis=0, keepdims=True)
        if self.M_type == "array":
            return tl.matmul(self.M.T, A*Csum0)
        else:
            return tl.matmul(self.M[1], tl.matmul(self.M[0].T, A*Csum0))
    def grad_fC(self, A, B, C):
        if self.M_type == "array":
            return (A*tl.matmul(self.M, B)).sum(axis=0, keepdims=True)
        else:
            return (tl.matmul(self.M[0].T, A) * tl.matmul(self.M[1].T, B)).sum(axis=0, keepdims=True)
    
    def e(self, A, B, C):
        if self.flag_epsilon:
            return 0
        else:
            return self.epsilon * sum([  self.nega_entropy(tmp) for tmp in (A, B, C) ])
    def grad_e(self, A, B, C):
        if self.flag_epsilon:
            return [0, 0, 0]
        else:
            return [ tl.log(tmp) * self.epsilon for tmp in (A, B, C)]
    
    def tau1KL(self, A, B, C):
        Bsum0 = B.sum(axis=0, keepdims=True)
        H1 = tl.matmul( A*Bsum0, tl.matmul(C.T, self.TL) )
        return self.tau1 * self.kl_div(H1, self.logL) + self.Lsum
    def grad_tau1KL(self, A, B, C):
        Bsum0 = B.sum(axis=0, keepdims=True)
        TLC = tl.matmul(self.TL.T, C)
        D1 = TLC * Bsum0
        H1 = tl.matmul(A, D1.T)
        grad_A = tl.matmul( (tl.log(H1)-tl.log(self.L)), D1)
        grad_B = ( TLC * tl.matmul((tl.log(H1)-tl.log(self.L)).T, A) ).sum(axis=0, keepdims=True)
        grad_C = tl.matmul(self.TL, tl.matmul((tl.log(H1)-tl.log(self.L)).T, A*Bsum0))
        if self.verbose:
            print(grad_A.shape, grad_B.shape, grad_C.shape)
        return [ tmp*self.tau1 for tmp in (grad_A, grad_B, grad_C)]
    
    def tau2KL(self, A, B, C):
        Asum0 = A.sum(axis=0, keepdims=True)
        H2 = tl.matmul( B*Asum0, tl.matmul(C.T, self.TR) )
        return self.tau2 * self.kl_div(H2, self.logR) + self.Rsum
    def grad_tau2KL(self, A, B, C):
        Asum0 = A.sum(axis=0, keepdims=True)
        TRC = tl.matmul(self.TR.T, C)
        D2 = TRC * Asum0
        H2 = tl.matmul(B, D2.T)
        grad_A = ( TRC * tl.matmul((tl.log(H2)-tl.log(self.R)).T, B) ).sum(axis=0, keepdims=True)
        grad_B = self.tau2 * tl.matmul( (tl.log(H2)-tl.log(self.R)), D2)
        grad_C = tl.matmul(self.TR, tl.matmul((tl.log(H2)-tl.log(self.R)).T, B*Asum0))
        if self.verbose:
            print(grad_A.shape, grad_B.shape, grad_C.shape)
        return [ tmp*self.tau1 for tmp in (grad_A, grad_B, grad_C)]

    def uAL(self, A, A_, B_, C_):
        Bsum0 = B_.sum(axis=0, keepdims=True)
        D1 = tl.matmul(self.TL.T, C_) * Bsum0
        D1sum0 = D1.sum(axis=0, keepdims=True)
        H1 = tl.matmul(A, D1.T)
        H1_ = tl.matmul(A_, D1.T)
        res = ((tl.log(A) - tl.log(A_))*A*D1sum0).sum() \
            + (H1*(tl.log(H1_)-self.logL-1)).sum()
        return res * self.tau1 + self.Lsum
    def grad_uAL(self, A, A_, B_, C_):
        Bsum0 = B_.sum(axis=0, keepdims=True)
        D1 = tl.matmul(self.TL.T, C_) * Bsum0
        D1sum0 = D1.sum(axis=0, keepdims=True)
        H1_ = tl.matmul(A_, D1.T)
        grad_A = tl.log(A/A_)*D1sum0 \
            + tl.matmul(tl.log(H1_)-self.logL, D1)
        return self.tau1 * grad_A
    def min_uAL(self, A_, B_, C_):
        Bsum0 = B_.sum(axis=0, keepdims=True)
        D1 = tl.matmul(self.TL.T, C_) * Bsum0
        D1sum0 = D1.sum(axis=0, keepdims=True)
        H1_ = tl.matmul(A_, D1.T)
        return A_ * tl.exp( tl.matmul(self.logL-tl.log(H1_), D1)/D1sum0 )
    
    def uAR(self, A, A_, B_, C_):
        Asum0 = A.sum(axis=0, keepdims=True)
        A_sum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR = tl.matmul(C_.T, self.TR)
        CRsum1 = CR.sum(axis=1)
        H2 = tl.matmul(B_*Asum0, CR)
        H2_ = tl.matmul(B_*A_sum0, CR)
        res = ((tl.log(A)-tl.log(A_))*(A*(Bsum0*CRsum1))).sum() \
            + (H2*(tl.log(H2_)-self.logR-1)).sum()
        return self.tau2 * res + self.Rsum
    def grad_uAR(self, A, A_, B_, C_):
        A_sum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR = tl.matmul(C_.T, self.TR)
        CRsum1 = CR.sum(axis=1)
        H2_ = tl.matmul(B_*A_sum0, CR)
        grad_A = tl.log(A/A_)*(Bsum0*CRsum1) \
               + (CR.T * tl.matmul((tl.log(H2_)-self.logR).T, B_)).sum(axis=0, keepdims=True)
        return self.tau2 * grad_A
    def min_uAR(self, A_, B_, C_):
        A_sum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR = tl.matmul(C_.T, self.TR)
        CRsum1 = CR.sum(axis=1)
        H2_ = tl.matmul(B_*A_sum0, CR)
        omega = (CR.T * tl.matmul((tl.log(H2_)-self.logR).T, B_)).sum(axis=0, keepdims=True)
        lamda = Bsum0*CRsum1
        return A_ * tl.exp(-omega/lamda)

    def uBL(self, B, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B.sum(axis=0, keepdims=True)
        B_sum0 = B_.sum(axis=0, keepdims=True)
        CL = tl.matmul(C_.T, self.TL)
        CLsum1 = CL.sum(axis=1)
        H1 = tl.matmul(A_*Bsum0, CL)
        H1_ = tl.matmul(A_*B_sum0, CL)
        res = ((tl.log(B)-tl.log(B_))*(B*(Asum0*CLsum1))).sum() \
            + (H1*(tl.log(H1_)-self.logL-1)).sum()
        return self.tau1 * res + self.Lsum
    def grad_uBL(self, B, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        B_sum0 = B_.sum(axis=0, keepdims=True)
        CL = tl.matmul(C_.T, self.TL)
        CLsum1 = CL.sum(axis=1)
        H1_ = tl.matmul(A_*B_sum0, CL)
        grad_B = tl.log(B/B_)*(Asum0*CLsum1) \
               + (CL.T * tl.matmul((tl.log(H1_)-self.logL).T, A_)).sum(axis=0, keepdims=True)
        return self.tau1 * grad_B
    def min_uBL(self, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        B_sum0 = B_.sum(axis=0, keepdims=True)
        CL = tl.matmul(C_.T, self.TL)
        CLsum1 = CL.sum(axis=1)
        H1_ = tl.matmul(A_*B_sum0, CL)
        omega = (CL.T * tl.matmul((tl.log(H1_)-self.logL).T, A_)).sum(axis=0, keepdims=True)
        lamda = Asum0*CLsum1
        return B_ * tl.exp(-omega/lamda)

        pass

    def uBR(self, B, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        D2 = tl.matmul(self.TR.T, C_) * Asum0
        D2sum0 = D2.sum(axis=0, keepdims=True)
        H2 = tl.matmul(B, D2.T)
        H2_ = tl.matmul(B_, D2.T)
        res = ((tl.log(B) - tl.log(B_))*B*D2sum0).sum() \
            + (H2*(tl.log(H2_)-self.logR-1)).sum()
        return res * self.tau2 + self.Rsum
    def grad_uBR(self, B, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        D2 = tl.matmul(self.TR.T, C_) * Asum0
        D2sum0 = D2.sum(axis=0, keepdims=True)
        H2_ = tl.matmul(B_, D2.T)
        grad_A = tl.log(B/B_)*D2sum0 \
            + tl.matmul(tl.log(H2_)-self.logR, D2)
        return self.tau1 * grad_A
    def min_uBR(self, B_, A_, C_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        D2 = tl.matmul(self.TR.T, C_) * Asum0
        D2sum0 = D2.sum(axis=0, keepdims=True)
        H2_ = tl.matmul(B_, D2.T)
        return B_ * tl.exp( tl.matmul(self.logR-tl.log(H2_), D2)/D2sum0 )


    def uCL(self, C, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CL = tl.matmul(C.T, self.TL)
        CL_ = tl.matmul(C_.T, self.TL)
        H1 = tl.matmul(A_*Bsum0, CL)
        H1_ = tl.matmul(A_*Bsum0, CL_)
        res = (tl.log(C/C_)*C*self.TLsum1*Asum0*Bsum0).sum() \
            + (H1*(tl.log(H1_)-self.logL-1)).sum()
        return self.tau1*res + self.Lsum
    def grad_uCL(self, C, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CL_ = tl.matmul(C_.T, self.TL)
        H1_ = tl.matmul(A_*Bsum0, CL_)
        tmp = tl.matmul( (tl.log(H1_)-self.logL).T, A_*Bsum0)
        grad_C = (tl.log(C/C_)*self.TLsum1*Asum0*Bsum0) \
               + tl.matmul(self.TL, tmp)
        return self.tau1 * grad_C
    def min_uCL(self, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CL_ = tl.matmul(C_.T, self.TL)
        H1_ = tl.matmul(A_*Bsum0, CL_)
        tmp = tl.matmul( (tl.log(H1_)-self.logL).T, A_*Bsum0)
        omega = tl.matmul(self.TL, tmp)
        lamda = self.TLsum1 * Asum0 * Bsum0
        return C_ * tl.exp(-omega/lamda)
    
    def uCR(self, C, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR = tl.matmul(C.T, self.TR)
        CR_ = tl.matmul(C_.T, self.TR)
        H2 = tl.matmul(B_*Asum0, CR)
        H2_ = tl.matmul(B_*Asum0, CR_)
        res = (tl.log(C/C_)*C*self.TRsum1*Asum0*Bsum0).sum() \
            + (H2*(tl.log(H2_)-self.logR-1)).sum()
        return self.tau2*res + self.Rsum
    def grad_uCR(self, C, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR_ = tl.matmul(C_.T, self.TR)
        H2_ = tl.matmul(B_*Asum0, CR_)
        tmp = tl.matmul( (tl.log(H2_)-self.logR).T, B_*Asum0)
        grad_C = (tl.log(C/C_)*self.TRsum1*Asum0*Bsum0) \
               + tl.matmul(self.TR, tmp)
        return self.tau2 * grad_C
    def min_uCR(self, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CR_ = tl.matmul(C_.T, self.TR)
        H2_ = tl.matmul(B_*Asum0, CR_)
        tmp = tl.matmul( (tl.log(H2_)-self.logR).T, B_*Asum0)
        omega = tl.matmul(self.TR, tmp)
        lamda = self.TRsum1 * Asum0 * Bsum0
        return C_ * tl.exp(-omega/lamda)

    def min_uCL_UCR(self, C_, A_, B_):
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        
        CL_ = tl.matmul(C_.T, self.TL)
        H1_ = tl.matmul(A_*Bsum0, CL_)
        tmp1 = tl.matmul( (tl.log(H1_)-self.logL).T, A_*Bsum0)

        CR_ = tl.matmul(C_.T, self.TR)
        H2_ = tl.matmul(B_*Asum0, CR_)
        tmp2 = tl.matmul( (tl.log(H2_)-self.logR).T, B_*Asum0)

        omega = self.tau1 * tl.matmul(self.TL, tmp1) + self.tau2 * tl.matmul(self.TR, tmp2)
        lamda = self.const_T * Asum0 * Bsum0
        return C_ * tl.exp(-omega/lamda)


    def F(self, A, B, C):
        return self.f(A, B, C) + self.e(A, B, C) + self.tau1KL(A, B, C) + self.tau2KL(A, B, C)
    def grad_F(self, A, B, C):
        grad_f = self.grad_f(A, B, C)
        grad_e = self.grad_e(A, B, C)
        grad_tau1KL = self.grad_tau1KL(A, B, C)
        grad_tau2KL = self.grad_tau2KL(A, B, C)
        return [grad_f[i]+grad_e[i]+grad_tau1KL[i]+grad_tau2KL[i] for i in range(3)]
    
    def GA(self, A, A_, B_, C_):
        return self.f(A, B_, C_) + self.e(A, B_, C_) \
              + self.uAL(A, A_, B_, C_) + self.uAR(A, A_, B_, C_)
    def grad_GA(self, A, A_, B_, C_):
        return self.grad_fA(A, B_, C_) + self.grad_e(A, B_, C_)[0] \
              + self.grad_uAL(A, A_, B_, C_) + self.grad_uAR(A, A_, B_, C_)
    def min_GA(self, A_, B_, C_):
        ## grad_f
        grad_fA = self.grad_fA(A_, B_, C_)
        
        ## grad_uAL
        Bsum0 = B_.sum(axis=0, keepdims=True)
        D1 = tl.matmul(self.TL.T, C_) * Bsum0
        D1sum0 = D1.sum(axis=0, keepdims=True)
        H1_ = tl.matmul(A_, D1.T)
        
        ## grad_uAR
        A_sum0 = A_.sum(axis=0, keepdims=True)
        #Bsum0 = B_.sum(axis=0, keepdims=True)
        CR = tl.matmul(C_.T, self.TR)
        CRsum1 = CR.sum(axis=1)
        H2_ = tl.matmul(B_*A_sum0, CR)

        omega = grad_fA \
              + self.tau1 * tl.matmul(tl.log(H1_)-self.logL, D1) \
              + self.tau2 * (CR.T * tl.matmul((tl.log(H2_)-self.logR).T, B_)).sum(axis=0, keepdims=True)
        lamda = D1sum0 + (Bsum0*CRsum1)
        if self.flag_epsilon:
            return A_ * tl.exp(-omega/lamda)
        else:
            scale = lamda + self.epsilon
            return (A_**(lamda/scale)) * tl.exp(-omega/scale)

    def GB(self, B, B_, A_, C_):
        return self.f(A_, B, C_) + self.e(A_, B, C_) \
              + self.uBL(B, B_, A_, C_) + self.uBR(B, B_, A_, C_)
    def grad_GB(self, B, B_, A_, C_):
        return self.grad_fB(A_, B, C_) + self.grad_e(A_, B, C_)[1] \
              + self.grad_uBL(B, B_, A_, C_) + self.grad_uBR(B, B_, A_, C_)
    def min_GB(self, B_, A_, C_):
        ## grad_f
        grad_fB = self.grad_fB(A_, B_, C_)

        ## grad_uBL
        Asum0 = A_.sum(axis=0, keepdims=True)
        B_sum0 = B_.sum(axis=0, keepdims=True)
        CL = tl.matmul(C_.T, self.TL)
        CLsum1 = CL.sum(axis=1)
        H1_ = tl.matmul(A_*B_sum0, CL)

        ## grad_uBR
        #Asum0 = A_.sum(axis=0, keepdims=True)
        D2 = tl.matmul(self.TR.T, C_) * Asum0
        D2sum0 = D2.sum(axis=0, keepdims=True)
        H2_ = tl.matmul(B_, D2.T)
        omega = grad_fB \
              + self.tau1 * (CL.T * tl.matmul((tl.log(H1_)-self.logL).T, A_)).sum(axis=0, keepdims=True) \
              + self.tau2 * tl.matmul(tl.log(H2_)-self.logR, D2)
        lamda = (Asum0*CLsum1) + D2sum0
        if self.flag_epsilon:
            return B_ * tl.exp(-omega/lamda)
        else:
            scale = lamda + self.epsilon
            return (B_**(lamda/scale)) * tl.exp(-omega/scale)

    def GC(self, C, C_, A_, B_):
        return self.f(A_, B_, C) + self.e(A_, B_, C) \
              + self.uCL(C, C_, A_, B_) + self.uCR(C, C_, A_, B_)
    def grad_GC(self, C, C_, A_, B_):
        return self.grad_fC(A_, B_, C) + self.grad_e(A_, B_, C) \
              + self.grad_uCL(C, C_, A_, B_) + self.grad_uCR(C, C_, A_, B_)
    def min_GC(self, C_, A_, B_):
        ## grad_f
        grad_fC = self.grad_fC(A_, B_, C_)

        ## grad_uCL
        Asum0 = A_.sum(axis=0, keepdims=True)
        Bsum0 = B_.sum(axis=0, keepdims=True)
        CL_ = tl.matmul(C_.T, self.TL)
        H1_ = tl.matmul(A_*Bsum0, CL_)
        tmp1 = tl.matmul( (tl.log(H1_)-self.logL).T, A_*Bsum0)

        ## grad_uCR
        #Asum0 = A_.sum(axis=0, keepdims=True)
        #Bsum0 = B_.sum(axis=0, keepdims=True)
        CR_ = tl.matmul(C_.T, self.TR)
        H2_ = tl.matmul(B_*Asum0, CR_)
        tmp2 = tl.matmul( (tl.log(H2_)-self.logR).T, B_*Asum0)
        
        omega = grad_fC \
              + self.tau1 * tl.matmul(self.TL, tmp1) \
              + self.tau2 * tl.matmul(self.TR, tmp2)
        lamda = tl.matmul(self.const_T, Asum0*Bsum0)
        if self.flag_epsilon:
            return C_ * tl.exp(-omega/lamda)
        else:
            scale = lamda + self.epsilon
            return (C_**(lamda/scale)) * tl.exp(-omega/scale)

    def nega_entropy(self, x):
        return (x*(tl.log(x)-1)).sum()
    
    def kl_div(self, x, logy):
        return (x*(tl.log(x)-logy-1)).sum()

    def get_device(self, device):
        if device == "numpy":
            device = "numpy"
        elif device == "pytorch":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "pytorch"
                else:
                    device = "numpy"
                    print("GPU is not avaliable, use numpy as default.")
            except:
                device = "numpy"
        else:
            raise ValueError("Only numpy or pytorch is avaliable.")
        return device

class MMOTNTD(OBJFUNC):
    """MMOTNTD

    Using BMM algorithm to solve the following problem:
     (A^*, B^*, C^*) = argmin <M, Adiag(C^T1)B^T>_F + \epsilon H(A,B,C) 
                          + \tau_1 KL(Adiag(B^T1)C^TT_L) + \tau_2 KL(Bdiag(A^T1)C^TT_R)
                    s.t. A >= \delta, B >= \delta, C >= \delta

    Parameters
    ----------
    L : (n_sender, n_ligand) numpy.ndarray
        The ligand expression matrix.
    R : (n_receiver, n_receptor) numpy.ndarray
        The receptor expression matrix.
    M : (n_sender, n_receiver) numpy.ndarray or tuple (M1, M2)
        The transport cost matrix computed using inverse optimal transport.
        If the input type is tuple, it should include two component that decomposed 
        from transport cost matrxi M.
    TL : (n_lr, n_ligand) numpy.ndarray
        The generated ligand coupling matrix (a binary matrix).
    TR : (n_lr, n_receptor) numpy.ndarray
        The generated ligand coupling matrix (a binary matrix).
    epsilon : int or float
        The coefficient of entropy regularization of tensor factor A, B, C, by default 0.
    tau1 : int or float
        The coefficient for KL divergence of unmatched ligand expression, by default 1.
    tau2 : int or float
        The coefficient for KL divergence of unmatched receptor expression, by default 1.
    delta : float
        The lower bound of tensor factor A, B, C, by default 1e-100.
    eps : float
        The lower bound of ligand or receptor expression, by default 1e-80.
    verbose : bool
        Whether to display the result of each iteration, by default False.
    log : bool
        Whether to save the result of each iteration, by default False.
    device : str, "numpy" or "pytorch"
        The backend used to perform this algorithm.

    Attributes
    ----------
    factors: tuple
        The solved result (A^*, B^*, C^*)
    
    """
    def __init__(self, L, R, M, TL, TR, epsilon=0, tau1=1, tau2=1, 
                 delta=1e-100, eps=1e-80, 
                 verbose=False, log=False, device="numpy"):
        super(MMOTNTD, self).__init__(L, R, M, TL, TR, epsilon, tau1, tau2, delta, eps,
                                       device, verbose, log)
    
    def random_init1(self, r, n_iter):
        N1, I_len = self.L.shape
        N2 = self.R.shape[0]
        if self.TL is not None:
            I_len = self.TL.shape[0]

        A0 = tl.random.random_tensor(shape=(N1, r))
        A0 = tl.where(A0>self.delta, A0, self.delta)
        B0 = tl.random.random_tensor(shape=(N2, r))
        B0 = tl.where(B0>self.delta, B0, self.delta)
        C0 = tl.random.random_tensor(shape=(I_len, r))
        C0 = tl.where(C0>self.delta, C0, self.delta)
        for _ in range(n_iter):
            Amin = self.min_uAR(A0, B0, C0)
            Bmin = self.min_uBL(B0, A0, C0)
            Cmin = (self.min_uCL(C0, A0, B0) + self.min_uCR(C0, A0, B0))/2

            A0 = tl.where(Amin>self.delta, Amin, self.delta)
            B0 = tl.where(Bmin>self.delta, Bmin, self.delta)
            C0 = tl.where(Cmin>self.delta, Cmin, self.delta)
        return A0, B0, C0
    
    def random_init2(self, r, n_iter1=10, n_iter2=50, n_round=100):
        if n_round == 0:
             optA, optB, optC = self.random_init1(r, n_iter=n_iter1)
        else:
            optF = np.Inf
            for _ in range(n_round):
                A_test, B_test, C_test = self.random_init1(r, n_iter=n_iter1)
                for _ in range(n_iter2):
                    C_test = self.min_GC(C_test, A_test, B_test)
                    C_test = tl.where(C_test>self.delta, C_test, self.delta)
                    A_test = self.min_GA(A_test, B_test, C_test)
                    A_test = tl.where(A_test>self.delta, A_test, self.delta)
                    B_test = self.min_GB(B_test, A_test, C_test)
                    B_test = tl.where(B_test>self.delta, B_test, self.delta)
                curF = self.F(A_test, B_test, C_test)
                if optF > curF:
                    optF = curF
                    optA, optB, optC = A_test.copy(), B_test.copy(), C_test.copy()
        return optA, optB, optC

    def get_device(self, device):
        if device == "numpy":
            device = "numpy"
        elif device == "pytorch":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "pytorch"
                else:
                    device = "numpy"
                    print("GPU is not avaliable, use numpy as default.")
            except:
                device = "numpy"
        else:
            raise ValueError("Only numpy or pytorch is avaliable.")
        return device 

    def fit(self, r, 
            init_val = None, init_params = [5, 50, 100],   ##KL, OT, n_round,
            block_order = "sequential", # or "random"
            min_iter=300, max_iter = 5000, stopThr = [1e-8, 1e-5], n_cpu = None,
            verbose=False, log=False):
        """

        Parameters
        ----------
        r : int
            Number of input rank
        init_val : None or tuple (A, B, C), optional
            The initial value of tensor factor A, B, C to iteration, by default None
        init_params : list[num1, num2, num3], optional
            The first number is the iteration step for KL divergence. The second numbder is the 
            iteration step for MMOTNTD. The third number is the round for repeat the above procedure.
            Set [0, 0, 0] for random initialization. By default [5, 50, 100]
        block_order : str "sequential" or "random", optional
            If "sequential", the order of block to update is C, B, A.
            If "random",randomize the order of coordinates in the BMM solver. 
            By default "sequential".
        min_iter : int, optional
            Minimum number of iterations, by default 300
        max_iter : int, optional
            Maximum number of iterations before timing out, by default 5000
        stopThr : list[num1, num2], optional
            The first number is absolute error of result between adjcent iteration.
            The second number is relative error of result between adjcent iteration.
            By default [1e-8, 1e-5].
        n_cpu : None or int, optional
            The number cpus to perform the iteration, by default None
        verbose : bool
            Whether to display the result of each iteration, by default False.
        log : bool
            Whether to save the result of each iteration, by default False.
        """
        
        ## reset parameters
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.stopThr = stopThr
        
        self.verbose = verbose
        self.log = log

        ## set n_cpu
        tl.set_backend(self.device)
        if self.device == "numpy" and n_cpu is not None:
            os.environ["OMP_NUM_THREADS"] = str(n_cpu)            # export OMP_NUM_THREADS=4
            os.environ["OPENBLAS_NUM_THREADS"] = str(n_cpu)       # export OPENBLAS_NUM_THREADS=4
            os.environ["MKL_NUM_THREADS"] = str(n_cpu)            # export MKL_NUM_THREADS=6
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cpu)     # export VECLIB_MAXIMUM_THREADS=4
            os.environ["NUMEXPR_NUM_THREADS"] = str(n_cpu)
        pass
        
        ## Initiation
        if init_val is None:
            A, B, C = self.random_init2(r, n_iter1=init_params[0], n_iter2=init_params[1], n_round=init_params[2])
        else:
            A = tl.tensor(init_val[0], device="cuda", dtype=tl.backend.float64)
            B = tl.tensor(init_val[1], device="cuda", dtype=tl.backend.float64)
            C = tl.tensor(init_val[2], device="cuda", dtype=tl.backend.float64)
        
        ## Update
        if block_order == "sequential":
            (Ae, Be, Ce) = self.sequential_update(A, B, C)
        elif block_order == "random":
            (Ae, Be, Ce) = self.random_update(A, B, C)

        self.factors = (Ae, Be, Ce)
    
    def rescale_tensor_factor(self, A, B, C):
        tmp = [A.sum(axis=0, keepdims=True), B.sum(axis=0, keepdims=True)]
        return A/tmp[0], B/tmp[1], C * (tmp[0] * tmp[1])

    def calc_obj_res(self, A, B, C):
        res = [ self.f(A,B,C), self.e(A, B, C), self.tau1KL(A, B, C), self.tau2KL(A, B, C) ]
        res.append(sum(res[:4]))
        return res

    def calc_var_err(self, var_prev, var_curr, method="max"):
        if method == "max":
            return [ tl.max(tl.abs(a-b)) for a, b in zip(var_prev, var_curr)]
        elif method == "norm2":
            return [ tl.norm(a-b, order=2) for a, b in zip(var_prev, var_curr)]
    
    def sequential_update(self, A0, B0, C0):
        A_test, B_test, C_test = A0.copy(), B0.copy(), C0.copy()
        prev_obj_val = self.F(A_test, B_test, C_test)
        
        if self.log:
            self.logmm = []
        
        N_continue, last_k = 0, 0
        for k in range(self.max_iter):
            #A_prev, B_prev, C_prev = A_test.copy(), B_test.copy(), C_test.copy()
            C_test = self.min_GC(C_test, A_test, B_test)
            C_test = tl.where(C_test>self.delta, C_test, self.delta)
            A_test = self.min_GA(A_test, B_test, C_test)
            A_test = tl.where(A_test>self.delta, A_test, self.delta)
            B_test = self.min_GB(B_test, A_test, C_test)
            B_test = tl.where(B_test>self.delta, B_test, self.delta)

            curr_obj_val = self.F(A_test, B_test, C_test)
            err = tl.abs(curr_obj_val-prev_obj_val)
            rel_err = err / tl.abs(curr_obj_val)
            #var_err1 = self.calc_var_err( (A_prev, B_prev, C_prev), (A_test, B_test, C_test), method="max")
            #var_err2 = self.calc_var_err( (A_prev, B_prev, C_prev), (A_test, B_test, C_test), method="norm2")
            prev_obj_val = curr_obj_val
            
            if self.log:
                self.logmm.append([curr_obj_val, rel_err, err])
            if self.verbose:
                print(k, curr_obj_val, rel_err, err, N_continue)

            if rel_err <= self.stopThr[0] and err <= self.stopThr[1]:
                if (k - last_k) == 1:
                    N_continue += 1
                else:
                    N_continue = 1
                last_k = k
            else:
                N_continue = 0
            if k>self.min_iter and N_continue >= 10:
                self.n_total_iter = k
                break
        self.n_total_iter = k
        return (A_test, B_test, C_test)
    
    def random_update(self, A0, B0, C0):
        A_test, B_test, C_test = A0.copy(), B0.copy(), C0.copy()
        prev_obj_val = self.F(A_test, B_test, C_test)

        if self.log:
            self.logmm = []

        N_continue, last_k = 0, 0
        n_block = 3
        #for k, idx in enumerate(np.random.randint(0, 3, size=self.max_iter*n_block)):
        order = np.array(range(n_block))
        for k in range(self.max_iter):
            np.random.shuffle(order)
            #A_prev, B_prev, C_prev = A_test.copy(), B_test.copy(), C_test.copy()
            for idx in order:
                if idx == 0:
                    A_test = self.min_GA(A_test, B_test, C_test)
                    A_test = tl.where(A_test>self.delta, A_test, self.delta)
                elif idx == 1:
                    B_test = self.min_GB(B_test, A_test, C_test)
                    B_test = tl.where(B_test>self.delta, B_test, self.delta)
                elif idx == 2:
                    C_test = self.min_GC(C_test, A_test, B_test)
                    C_test = tl.where(C_test>self.delta, C_test, self.delta)                        

            curr_obj_val = self.F(A_test, B_test, C_test)
            err = tl.abs(curr_obj_val-prev_obj_val)
            rel_err = err / tl.abs(curr_obj_val)
            #var_err1 = self.calc_var_err( (A_prev, B_prev, C_prev), (A_test, B_test, C_test), method="max")
            #var_err2 = self.calc_var_err( (A_prev, B_prev, C_prev), (A_test, B_test, C_test), method="norm2")

            prev_obj_val = curr_obj_val

            if self.log:
                self.logmm.append([curr_obj_val, rel_err, err])
            if self.verbose:
                print(k, curr_obj_val, rel_err, err, N_continue)

            if rel_err <= self.stopThr[0] and err <= self.stopThr[1]:
                if (k - last_k) == 1:
                    N_continue += 1
                else:
                    N_continue = 1
                last_k = k
            else:
                N_continue = 0
            if k>self.min_iter and N_continue >= 10:
                self.n_total_iter = k
                break
        self.n_total_iter = k
        return (A_test, B_test, C_test)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from simulate import generate_coupling_matrix, simu_data

    ## simu data
    r, N1, N2, I_len = 20, 123, 124, 1000                 # A, B, C
    m = generate_coupling_matrix(I_len, 400)              # TL, L
    n = generate_coupling_matrix(I_len, 300)              # TR, R
    K, epsilon = 1e4, 1                                   # M
    is_decom, d, solver = True, 10, "svd"                 # M1, M2

    print("### Simulation test")
    r, A, B, C, L, R, TL, TR, M, (M1, M2) = simu_data(r, N1, N2, I_len, m, n, K, epsilon, is_decom, d, solver)
    err = np.abs(M - np.dot(M1, M2.T))
    print("True data", r, A.shape, B.shape, C.shape, sep="\t")
    print("Simu data", L.shape, R.shape, TL.shape, TR.shape, M.shape, M1.shape, M2.shape)
    print("Error", err.sum(), err.max())
    #"""

    ## run: MMOTNTF
    mmot = MMOTNTD(L, R, M, TL, TR, epsilon=0.01, tau1=1, tau2=1, delta=1e-20, verbose=False)
    mmot.fit(r=20, init_params=[0, 0, 0], min_iter=0, max_iter=20)
    Ae, Be, Ce = mmot.factors
    res = mmot.calc_obj_res(Ae, Be, Ce)
    print( res )

    mmot = MMOTNTD(L, R, M, TL, TR, epsilon=0, tau1=1, tau2=1, delta=1e-20, verbose=False)
    mmot.fit(r=20, init_params=[0, 0, 0], min_iter=0, max_iter=20)
    Ae, Be, Ce = mmot.factors
    res = mmot.calc_obj_res(Ae, Be, Ce)
    print( res )

    mmot = MMOTNTD(L, R, (M1, M2), TL, TR, epsilon=0.01, tau1=1, tau2=1, delta=1e-20, verbose=False)
    mmot.fit(r=20, init_params=[0, 0, 0], min_iter=0, max_iter=20)
    Ae, Be, Ce = mmot.factors
    res = mmot.calc_obj_res(Ae, Be, Ce)
    print( res )

    mmot = MMOTNTD(L, R, (M1, M2), TL, TR, epsilon=0, tau1=1, tau2=1, delta=1e-20, verbose=False)
    mmot.fit(r=20, init_params=[0, 0, 0], min_iter=0, max_iter=20)
    Ae, Be, Ce = mmot.factors
    res = mmot.calc_obj_res(Ae, Be, Ce)
    print( res )

