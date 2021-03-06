import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from solvers import RK4step, FEstep
from qm import stepEvolution, decimalToBinary, plot
from hamiltonians import Grover
from matplotlib.animation import FuncAnimation

threads = 1
target = [2, 4, 7]
N = 3
H = Grover

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
def update(t):
    global lines
    stepEvolution(phiplus, params, dt, RK4step)
    lines = plot(fig, ax, phiplus, N, '|psi>', lines, t/runtime, runtimetitle)
    return lines[0], ax

anim = FuncAnimation(fig, update, frames=lin, interval=50, repeat=False)
# if you want to save as gif
#anim.save('name.gif', dpi=100, writer='imagemagick')
# if you want to watch it evolve live
plt.show()
