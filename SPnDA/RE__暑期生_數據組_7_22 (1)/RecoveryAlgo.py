import cvxpy as cvx 
import numpy as np

import scipy.optimize.root

__author__ = "Jean-Luc Bouchot"
__copyright__ = "Copyright 2019, School of Mathematics and Statistics, Beijing Institute of Technology --- For educational uses only"
__credits__ = "Jean-Luc Bouchot"
__license__ = "GPL"
__version__ = "0.1.0-dev"
__maintainer__ = "Jean-Luc Bouchot"
__email__ = "jlbouchot@gmail.com"
__status__ = "Development"
__lastmodified__ = "2019/07/19"



def DFT_matrix(N): # One could also generate the DFT matrix via scipy.linalg.dft
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W


def BasisPursuit(y,A,eps = 1e-5): 
    [m,N] = A.shape
    
    x = cvx.Variable(N)
    
    cvx.Problem(cvx.Minimize(cvx.norm(x, 1)),
        [cvx.norm2(cvx.matmul(A,x) - y) <= eps]).solve(solver=cvx.SCS)
    return x.value
