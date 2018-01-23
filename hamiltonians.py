import numpy as np
from qm import stepEvolution, decimalToBinary

def Grover(phik, beg, end, N, D, target) :
    phim = np.zeros(end - beg, np.complex64)
    for m in range(D) :
        for n in range(beg, end) :
            s = 1./D
            if m == n and n in target :
                s += 1.
            phim[n - beg] += phik[m]*s
    phim *= -1j
    return beg, end, phim
