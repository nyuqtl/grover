import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from solvers import RK4step, FEstep
from qm import stepEvolution, decimalToBinary, plot
from matplotlib.animation import FuncAnimation

threads = 2
target = [2, 7]
N = 3

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

runtime = np.sqrt(np.float64(D)/np.float64(M))*np.pi
runtimetitle = 'sqrt(D/M)*pi'

phiplus = np.zeros(2**N, np.complex64)
for i in range(D) :
    phiplus[i] = 1./D

np.seterr(invalid='raise')

fig, ax = plt.subplots(2,1)
global lines
lines = plot(fig, ax, phiplus, N, '|psi>', [], 0.0, runtimetitle)
lin, dt = np.linspace(0, runtime, 100, retstep=True)

params = (threads, N, D, target, H)
stepEvolution(phiplus, params, dt, RK4step)
def update(t):
    global lines
    stepEvolution(phiplus, params, dt, RK4step)
    lines = plot(fig, ax, phiplus, N, '|psi>', lines, t/runtime, runtimetitle)
    return lines[0], ax

anim = FuncAnimation(fig, update, frames=lin, interval=50, repeat=False)
# if you want to save as gif
#anim.save('dissip1equalsuperpos.gif', dpi=100, writer='imagemagick')
# if you want to watch it evolve live
plt.show()
