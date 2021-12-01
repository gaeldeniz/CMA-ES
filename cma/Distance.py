import numpy as np
from numpy import array, dot, isscalar, sum
import math
from scipy.linalg import sqrtm
from pycma import cma


class distance:
    def __init__(self, es, func, type):
        if type == 'optimal':
            #es_optimal = cma.CMAEvolutionStrategy(**es.inputargs)
            #es_optimal.optimize(func)
            xopt, es_optimal = cma.fmin2(func, es.inputargs['x0'], es.inputargs['sigma0'], {'verb_disp' : 0})
            self.optimal_C = es_optimal.C
        elif type == 'identity':
            self.optimal_C = np.identity(len(es.inputargs['x0']))
        return 
    def distance(self, C):
        inv_H = np.linalg.inv(self.optimal_C)
        square_root_H = sqrtm(inv_H)
        M = square_root_H @ C @ square_root_H

        eigenvalues_M, eigenvectors_M = np.linalg.eig(M)

        mean_log_eigenvalues_M = np.mean(np.log(eigenvalues_M))

        distance = 0
        for eigenvalue in eigenvalues_M:
            distance += (math.log(eigenvalue) - mean_log_eigenvalues_M)**2

        return math.sqrt(distance)
    