import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from solvers import RK4step
from qm import stepEvolution, decimalToBinary

threads = 1
target = [2, 7, 11, 22, 28, 31, 44, 51, 59, 63]
N = 6

print 'N = ' + str(N)
print 'target = ' + str(target)

# Hamiltonian
def H(phik, beg, end, N, D, target) :
    phim = np.zeros(end - beg, np.complex64)
    for m in range(D) :
        for n in range(beg, end) :
            s = 1./D
            if m == n and n in target :
                s += 1.
            phim[n - beg] += phik[m]*s
    phim *= -1j
    return beg, end, phim

# number of spin configurations
D = 2**N
M = len(target)
tol = 1e-6
dt = 1e-2

runtime = np.sqrt(np.float64(D)/np.float64(M))*np.pi/2.

psi = np.zeros(2**N, np.complex64)
for i in range(D) :
    psi[i] = 1./D
params = (threads, N, D, target, H)

print '\nPerforming time evolution to check if target states will get amplified'
print '(this might take a while)'

total = 0.
while total < runtime :
    dtt = dt
    if total + dt > runtime :
        dtt = runtime - total
    stepEvolution(psi, params, dtt, RK4step)
    total += dtt

expectation = np.power(np.abs(psi), 2.)
outside = None

for i in range(D) :
    if i in target :
        if not np.abs(1./len(target) - expectation[i]) < tol :
            raise ValueError('Target space not evenly distributed at final stage')
    else :
        if outside is None :
            outside = i
        if not expectation[i] < tol :
            raise ValueError('States outside of target space not sufficiently close to zero')

print '\nStates in target space sufficiently close to ' + str(1./len(target)) + ', (' + str(expectation[target[0]]) + ')'

print '\nStates outside of target space sufficiently close to 0., (' + str(expectation[outside]) + ')'

print '\nPerforming another time evolution to check if system returns to equal superposition'
print '(this might take a while)'

total = 0.
while total < runtime :
    dtt = dt
    if total + dt > runtime :
        dtt = runtime - total
    stepEvolution(psi, params, dtt, RK4step)
    total += dtt

expectation = np.power(np.abs(psi), 2.)

for i in range(D) :
    if not np.abs(1./D - expectation[i]) < tol :
        raise ValueError('\nAfter return states are not close enough to even superposition')

print '\nAfter another time evolution states returned to even superposition'
