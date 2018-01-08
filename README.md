## Introduction

This is a draft of program that is used for simulation of quantum algorithms, here we test is on Grover's algorithm. It is a continuous simulation that integrates Schrodinger's equation over time using Runge-Kutta method. It is not a discrete, unitary simulation.

![example](https://github.com/nyuqtl/grover/blob/master/mod1.gif?raw=true "example")

## Features

* Parallel processing, `O(2^n^2^n)` to `O(2^n^(2^n)/k)` where `k` is number of available threads
* RK4 integrator (and having it easy to add more integrators, just look up [solvers.py](./solvers.py))
* Easy to change Hamiltonian, simply write your own Hamiltonian evaluation function, example in [run.py](./run.py#L15-L32)
* Possibility of running the simulation numerically with graph updated in real-time (no need to wait until entire simulation is done to generate animated plot)
