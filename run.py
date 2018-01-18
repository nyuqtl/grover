import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from solvers import RK4step, FEstep
from qm import stepEvolution, decimalToBinary, plot
from matplotlib.animation import FuncAnimation

threads = 1
target = [2, 7]
N = 3

# Hamiltonian
def H(phik, beg, end, N, D, target) :
    phim = np.zeros(end - beg, np.complex64)
    for m in range(beg, end) :
        E0 = 0.
        E1 = 1.
        # source states Hadamard
        Ps = 0.
        if m == 0 :
            Ps = phik[0]
        Pt = 0.
        if m in target :
            mbin = decimalToBinary(m, N)
            for n in range(0, D) :
                nbin = decimalToBinary(n, N)
                Pt += (phik[n]*np.power(-1., np.dot(nbin, mbin)))/np.sqrt(D)
        phim[m - beg] = -1j*(E1*(Ps + Pt) + E0*(D - len(target)))
    return beg, end, phim

# number of spin configurations
D = 2**N
M = len(target)

runtime = np.sqrt(np.float64(M)/np.float64(D))*np.pi
runtimetitle = 'sqrt(M/D)*pi'

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
#anim.save('dissip1.gif', dpi=100, writer='imagemagick')
# if you want to watch it evolve live
plt.show()
