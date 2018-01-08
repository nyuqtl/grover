import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from solvers import RK4step, FEstep
from qm import stepEvolution, decimalToBinary, plot
from matplotlib.animation import FuncAnimation

threads = 2
target = [7]
N = 4

# Hamiltonian
def H(phik, beg, end, N, D, target) :
    phim = np.zeros(end - beg, np.complex64)
    for m in range(beg, end) :
        res = 1.
        # Grover
        if m == 0 :
            res = 0.
        s1 = 0.
        mbin = decimalToBinary(m, N)
        for n in range(D) :
            nbin = decimalToBinary(n, N)
            # Oracle
            orc = 1.
            if n in target :
                orc = 0.
            s1 += np.power(-1., np.dot(mbin, nbin))*orc*phik[n]
        phim[m - beg] = res*(s1)*-1j
    return beg, end, phim

# number of spin configurations
D = 2**N

runtime = np.sqrt(D)*np.pi
runtimetitle = 'sqrt(D)*pi'

phiplus = np.zeros(2**N, np.complex64)
phiplus[0] = 1.

np.seterr(invalid='raise')

fig, ax = plt.subplots(2,1)
global lines
lines = plot(fig, ax, phiplus, N, '|psi>', [], 0.0, runtimetitle)
lin, dt = np.linspace(0, runtime, 100, retstep=True)

params = (threads, N, D, target, H)
def update(t):
    global lines
    stepEvolution(phiplus, params, dt, RK4step)
    lines = plot(fig, ax, phiplus, N, '|psi>', lines, t/runtime, runtimetitle)
    return lines[0], ax

anim = FuncAnimation(fig, update, frames=lin, interval=50, repeat=False)
# if you want to save as gif
#anim.save('example.gif', dpi=130, writer='imagemagick')
# if you want to watch it evolve live
plt.show()
