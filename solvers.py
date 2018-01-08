import numpy as np
import sys

def FEstep(x, t, h, f, params) :
    arguments = tuple([x, t] + list(params))
    return h*f(*arguments)

def RK2step(x, t, h, f, params) :
    k1 = FEstep(x, t, h, f, params)
    k2 = FEstep(x+0.5*k1, t+0.5*h, h, f, params)
    return k2

def RK4step(x, t, h, f, params) :
    k1 = FEstep(x, t, h, f, params)
    k2 = FEstep(x+0.5*k1, t+0.5*h, h, f, params)
    k3 = FEstep(x+0.5*k2, t+0.5*h, h, f, params)
    k4 = FEstep(x+k3, t+h, h, f, params)
    return np.divide(k1+2*k2+2*k3+k4,6.0)
