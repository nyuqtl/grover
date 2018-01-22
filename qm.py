import sys
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Pool

def binaryToDecimal(configuration) :
    return int('0b'+''.join([str(x) for x in configuration]), 2)

def decimalToBinary(number, N) :
    binary = [int(x) for x in bin(number)[2:]]
    while len(binary) < N :
        binary = [0] + binary
    return binary

def normChunk(phi, beg, end) :
    norm = np.complex64(0.)
    for i in range(beg, end) :
        norm += (phi[i]*phi[i].conjugate()).real
    return norm

def applyNormChunk(phi, norm, beg, end) :
    for i in range(beg, end) :
        phi[i] *= np.complex64(1./np.sqrt(norm))
    return beg, end, phi[beg:end]

def normalize(phi, D, threads) :
    if threads == 1:
        norm = normChunk(phi, 0, D)
        return applyNormChunk(phi, norm, 0, D)
    chunk = D / threads
    p = Pool(threads)
    results = []
    for t in range(threads) :
        beg = t*chunk
        end = t*chunk+chunk
        results.append(p.apply_async(normChunk, (phi, beg, end)))
    norm = np.complex64(0.)
    for res in results :
        norm += res.get()
    results = []
    for t in range(threads) :
        beg = t*chunk
        end = t*chunk+chunk
        results.append(p.apply_async(applyNormChunk, (phi, norm, beg, end)))
    for res in results :
        (beg, end, arr) = res.get()
        phi[beg:end] = arr
    p.close()
    return phi

def stepEvolution(phi, params, dt, step) :
    threads, N, D, target, hamiltonian = params
    phi += step(phi, 0., dt, evaluate, params)
    return normalize(phi, D, threads)

def plot(fig, ax, phi, N, title, lines, time, total) :
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    fig.suptitle(title + ' at ' + ("%.2f" % time) + '*' + total, fontsize=12)
    states = np.arange(2**N)
    ax[0].set_ylabel('Psi')
    ax[1].set_ylabel('Expectation')
    if len(lines) == 0 :
        l1 = ax[0].bar(states - 0.15, phi.imag, 0.30, label='Imaginary')
        l2 = ax[0].bar(states + 0.15, phi.real, 0.30, label='Real')
        l3 = ax[1].bar(states,np.power(np.abs(phi), 2.0))
        lines = [l1, l2, l3]
    else :
        expect = np.power(np.abs(phi), 2.0)
        for i in states :
            lines[0][i].set_height(phi[i].imag)
            lines[1][i].set_height(phi[i].real)
            lines[2][i].set_height(expect[i])
    NN = 2**N - 1
    for i in range(2) :
        ax[i].get_yaxis().set_label_coords(-0.1,0.5)
        ax[i].grid(True)
        ax[i].set_xticks(states)
        ax[i].set_ylim([-1., 1.])
    ax[1].set_ylim([0., 1.])
    ax[1].set_yticks([0., 0.25, 0.5, 0.75, 1.])
    return lines

def evaluate(phik, T, threads, N, D, target, hamiltonian) :
    phim = np.zeros(D, np.complex64)
    if threads == 1 :
        phim = hamiltonian(phik, 0, D, N, D, target)
        return phim
    chunk = D / threads
    p = Pool(threads)
    results = []
    for t in range(threads) :
        beg = t*chunk
        end = t*chunk+chunk
        results.append(p.apply_async(hamiltonian, (phik, beg, end, N, D, target)))
    for res in results :
        (beg, end, arr) = res.get()
        phim[beg:end] = arr
    p.close()
    return phim

def measure(q, N, D, phi) :
    phim = np.zeros(D, np.complex64)
    # accumulated probability for spin up and spin down resp.
    total = [0., 0.]
    for i in range(D) :
        # get spin configuration of phi[i]
        conf = decimalToBinary(i, N)
        # what is the spin orientation for qubit q on state phi[i]
        r = conf[q]
        # get probability of obtaining phi[i]
        p = (phi[i]*phi[i].conjugate()).real
        # update total probability
        total[r] += p
    # quantum system represents a fair coin flip
    select = random.random()
    measurement = 0
    if select >= total[0] :
        measurement = 1
    # remove terms inconsistent with measurement from the quantum state
    # and renormalize
    scalar = np.sqrt(1./total[measurement])
    for i in range(D) :
        conf = decimalToBinary(i, N)
        # keep only terms that are consistent with measurement
        if conf[q] == measurement :
            phim[i] = phi[i]*scalar
    # measurement influenced quantum state causing information loss, phi became phim
    return measurement, phim

def measureTotalProbParallel(q, N, phi, beg, end) :
    total = [0., 0.]
    for i in range(beg, end) :
        conf = decimalToBinary(i, N)
        r = conf[q]
        p = (phi[i]*phi[i].conjugate()).real
        total[r] += p
    return total[0], total[1]

def measureRenormalizeParallel(q, N, phi, measurement, scalar, beg, end) :
    phim = np.zeros(end - beg, np.complex64)
    for i in range(beg, end) :
        conf = decimalToBinary(i, N)
        if conf[q] == measurement :
            phim[i - beg] = phi[i]*scalar
    return beg, end, phim

def measureParallel(q, N, D, phi, threads) :
    p = Pool(threads)
    chunk = D / threads
    phim = np.zeros(D, np.complex64)
    total = [0., 0.]
    results = []
    for t in range(threads) :
        beg = t*chunk
        end = t*chunk+chunk
        results.append(p.apply_async(measureTotalProbParallel, (q, N, phi, beg, end)))
    for res in results :
        p0, p1 = res.get()
        total[0] += p0
        total[1] += p1
    select = random.random()
    measurement = 0
    if select >= total[0] :
        measurement = 1
    scalar = np.sqrt(1./total[measurement])
    results = []
    for t in range(threads) :
        beg = t*chunk
        end = t*chunk+chunk
        results.append(p.apply_async(measureRenormalizeParallel, (q, N, phi, measurement, scalar, beg, end)))
    for res in results :
        (beg, end, arr) = res.get()
        phim[beg:end] = arr
    p.close()
    return measurement, phim
