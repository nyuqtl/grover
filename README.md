## Introduction

This is a draft of program that is used for simulation of quantum algorithms, here we test is on Grover's algorithm. It is a continuous simulation that integrates Schrodinger's equation over time using Runge-Kutta method. It is not a discrete, unitary simulation.

single target state

![example](https://github.com/nyuqtl/grover/blob/master/animations/single.gif?raw=true "example")

multiple target states

![example](https://github.com/nyuqtl/grover/blob/master/animations/multi.gif?raw=true "example")

## Features

* Parallel processing, `O(2^n^2^n)` to `O(2^n^(2^n)/k)` where `k` is number of available threads
* RK4 integrator (and having it easy to add more integrators, just look up [solvers.py](./solvers.py))
* Easy to change Hamiltonian, simply write your own Hamiltonian evaluation function, example in [run.py](./run.py#L15-L32)
* Possibility of running the simulation numerically with graph updated in real-time (no need to wait until entire simulation is done to generate animated plot)
* Export animations to a file (requires `imagemagick`)

## Test script

We prepared a simple tool that allows to check if after `sqrt(D/M)*pi/2` time evolution reaches even superposition of only target states and after evolving for same time again if returns to initial state. Simply run [large.py](./large.py)

Result should be

```
N = 6
target = [2, 7, 11, 22, 28, 31, 44, 51, 59, 63]

Performing time evolution to check if target states will get amplified
(this might take a while)

States in target space sufficiently close to 0.1, (0.1)

States outside of target space sufficiently close to 0., (2.29956e-15)

Performing another time evolution to check if system returns to equal superposition
(this might take a while)

After another time evolution states returned to even superposition
```
