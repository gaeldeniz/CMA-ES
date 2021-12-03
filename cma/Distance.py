import numpy as np
from numpy import array, dot, isscalar, sum
import math
from scipy.linalg import sqrtm
from pycma import cma
import matplotlib.pyplot as plt

class distance:
    def __init__(self, es, func, type, Hessian = None):
        self.type = type
        self.logger = []
        if type == 'optimal':
            #es_optimal = cma.CMAEvolutionStrategy(**es.inputargs)
            #es_optimal.optimize(func)
            xopt, es_optimal = cma.fmin2(func, es.inputargs['x0'], es.inputargs['sigma0'], {'verb_disp' : 0})
            self.optimal_C = es_optimal.C
        elif type == 'identity':
            self.optimal_C = np.identity(len(es.inputargs['x0']))
        elif type == 'known':
            self.hessian =  Hessian
        return 
    def distance(self, C):
        if self.type == 'known':
            Hes = self.hessian
        else:
            Hes = np.linalg.inv(self.optimal_C)
        square_root_H = sqrtm(Hes)
        M = square_root_H @ C @ square_root_H

        eigenvalues_M, eigenvectors_M = np.linalg.eig(M)

        if not np.all(eigenvalues_M):
            raise ValueError("The hessian matrix provided generates eigen values which are not compatible with the distance metric. Please try a different type of distance metric")

        mean_log_eigenvalues_M = np.mean(np.log(eigenvalues_M))

        distance = 0
        for eigenvalue in eigenvalues_M:
            distance += (math.log(eigenvalue) - mean_log_eigenvalues_M)**2
        distance = math.sqrt(distance)
        self.logger.append(distance)
        return distance
    
    def plot(self):
        plt.plot(self.logger)
        plt.xlabel('iteration')
        plt.ylabel('Distance')
        if self.type == "known":
            plt.title('Distance to Known Hessian at the optimal')
        elif self.type == "optimal":
            plt.title('Distance to Approximate Hessian at the optimal')
        else:
            plt.title('Distance to Identity matrix')
        plt.grid()
        plt.show()
    