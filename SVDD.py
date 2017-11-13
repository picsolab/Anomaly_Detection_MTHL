# @Support Vector Data Description - SVDD
# @Xian Teng
# @2017
# @References:
#   [1] Tax et al, Support Vector Data Description, 2004
#   [2] Rong et al, "Working Set Selection Using Second Order Information for Training Support Vector Machines", 2005

"""
-----------------------------------------------
Resolve Quadratic optimization problem as below:
    min_{alpha} 0.5*alpha.T*Q*alpha+p.T*alpha
    subject to
        0 <= alpha[i] <= Cp
        sum_i alpha[i] = 1.0
-----------------------------------------------
"Decomposition method" updates a subset of alpha values per iteration, and the subset is denoted as "working set",
We use one special decomposition method called "Sequential Minimal Optimization (SMO)", which restricts working set
to have only two elements. To select a working set, we refer to ref[2] using second order information. The code is
developed by following Appendix B. Pseudo Code for WSS 3 (p1896) and Algorithm 2 (p1896) and Theorem 3 (p1894).
"""
import numpy as np

class Kernel(object):
    """in our case, kernel is Tr(x*y.T)"""
    @staticmethod
    def trace():
        return lambda x, y: np.trace(np.dot(x,np.transpose(y)))

class SVDDTrainer(object):
    def __init__(self, kernel, Cp):
        self._kernel = kernel # take kernel function as argument
        self._Cp = Cp # 
        self._tau = 1e-12 # a very small positive number
        self._eps = 1e-3 # tolerance

    def _gram_matrix(self,YY):
        KM = np.array([[self._kernel(YY[i],YY[j]) for i in range(len(YY))] for j in range(len(YY))])
        return KM

    def _secondOrder_select_working_set(self,alpha,G,Q,n_samples):
        i = -1
        Gmax = -float("inf")
        Gmin = float("inf")
        for t in np.arange(n_samples):
            if alpha[t] < self._Cp:
                if -G[t] > Gmax:
                    Gmax = -G[t]
                    i = t

        j = -1
        obj_diff_min = float("inf")
        for t in np.arange(n_samples):
            if alpha[t] > 0:
                b = Gmax + G[t]
                if -G[t] < Gmin:
                    Gmin = -G[t]

                if b > 0:
                    a = Q[i,i] + Q[t,t] - 2 * Q[i,t]
                    if a < 0:
                        a = self._tau
                    obj_diff = -(b*b)/a
                    if obj_diff <= obj_diff_min:
                        j = t
                        obj_diff_min = obj_diff

        if Gmax - Gmin < self._eps or j == -1:
            return (-1,-1)
        else:
            return (i,j)



    # min f = 0.5 * alpha.T * Q * alpha + p.T * alpha
    # gradient of f is G = Q * alpha + p
    def train(self,X,tau):
        # calculate quadratic matrix Q and vector p
        n_samples = len(tau)
        KM = self._gram_matrix(X)
        Q = KM
        p = [np.sum([KM[i,j]*tau[j] for j in range(n_samples)])-1.0*KM[i,i] for i in range(n_samples)]

        # initialize alpha
        alpha = np.ones(n_samples)
        alpha = alpha / sum(alpha)

        # initialize grad_f which is denoted as G
        G = np.sum(np.array([alpha[i]*Q[:,i] for i in range(n_samples)]),axis=0)
        G = G + p

        # calculating alpha(s)
        cnt = 0
        while(1):
            # i,j,exit = self._firstOrder_select_working_sets(alpha,G,n_samples)
            i,j = self._secondOrder_select_working_set(alpha,G,Q,n_samples)
            if j == -1:
                break

            cnt = cnt + 1

            a = Q[i,i] + Q[j,j] - 2 * Q[i,j]
            if a <= 0:
                a = self._tau
            b = -G[i]+G[j]

            # update alpha(s)
            old_alpha_i = alpha[i]
            old_alpha_j = alpha[j]
            # delta = (-G[i]+G[j])/max(KM[i,i] + KM[j,j] -2*KM[i,j], 0)
            delta = b/a
            alpha[i] += delta
            alpha[j] -= delta
            
            const = old_alpha_i + old_alpha_j
            if alpha[i] < 0:
                alpha[i] = 0
            elif alpha[i] > self._Cp:
                alpha[i] = self._Cp

            alpha[j] = const - alpha[i]
            if alpha[j] < 0:
                alpha[j] = 0
            elif alpha[j] > self._Cp:
                alpha[j] = self._Cp

            alpha[i] = const - alpha[j]

            # update gradient
            delta_alpha_i = alpha[i] - old_alpha_i
            delta_alpha_j = alpha[j] - old_alpha_j
            G += Q[:,i] * delta_alpha_i + Q[:,j] * delta_alpha_j
        return alpha