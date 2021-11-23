import numpy as np
from numpy import array, dot, isscalar, sum
import math
from scipy.linalg import sqrtm
from pycma import cma


class distance:
    def __init__(self, func):
        if func == cma.ff.rosen:
            print("Function is Rosen")
            self.optimal_C = np.array([[4.47586147e-05, 1.50437015e-05, 1.04809432e-05, 1.12687443e-05,
                                    1.98053130e-05, 5.42267933e-05, 1.13281263e-04, 2.29251212e-04],
                                [1.50437015e-05, 4.89920424e-05, 4.55584007e-05, 4.48289335e-05,
                                    7.02914983e-05, 1.34456913e-04, 2.73247170e-04, 5.51588562e-04],
                                [1.04809432e-05, 4.55584007e-05, 1.21089652e-04, 1.11075173e-04,
                                    1.70610815e-04, 3.27828645e-04, 6.84982821e-04, 1.37914695e-03],
                                [1.12687443e-05, 4.48289335e-05, 1.11075173e-04, 1.74321532e-04,
                                    2.70645187e-04, 5.11131865e-04, 1.04008197e-03, 2.07967590e-03],
                                [1.98053130e-05, 7.02914983e-05, 1.70610815e-04, 2.70645187e-04,
                                    5.46048207e-04, 1.00445141e-03, 2.05292194e-03, 4.10535896e-03],
                                [5.42267933e-05, 1.34456913e-04, 3.27828645e-04, 5.11131865e-04,
                                    1.00445141e-03, 1.98638515e-03, 4.07447879e-03, 8.14133137e-03],
                                [1.13281263e-04, 2.73247170e-04, 6.84982821e-04, 1.04008197e-03,
                                    2.05292194e-03, 4.07447879e-03, 8.47118809e-03, 1.69441401e-02],
                                [2.29251212e-04, 5.51588562e-04, 1.37914695e-03, 2.07967590e-03,
                                    4.10535896e-03, 8.14133137e-03, 1.69441401e-02, 3.40749906e-02]])
            self.optimal_C = np.identity(8)
            print(self.optimal_C)
        else:
            self.optimal_C = np.identity(8)    

    def distance(self, C):
        square_root_H = sqrtm(self.optimal_C)
        M = square_root_H @ C @ square_root_H

        eigenvalues_M, eigenvectors_M = np.linalg.eig(M)

        mean_eigenvalues_M = np.mean(eigenvalues_M)

        #difference_eigenvalues = eigenvalues_M -mean_eigenvalues_M

        distance = 0
        for eigenvalue in eigenvalues_M:
            distance += math.log(eigenvalue) - math.log(mean_eigenvalues_M)

        return distance