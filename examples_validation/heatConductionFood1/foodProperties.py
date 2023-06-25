
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def kPro(t):
    a = 0.17881
    b = 0.0011958
    c = -0.0000027178
    return a + b * t + c * t**2

def kFat(t):
    a = 0.18071
    b = -0.00027604
    c = -0.00000017749
    return a + b * t + c * t**2

def kCar(t):
    a = 0.20141
    b = 0.0013874
    c = -0.0000043312
    return a + b * t + c * t**2

def kFib(t):
    a = 0.18331
    b = 0.0012497
    c = -0.0000031683
    return a + b * t + c * t**2

def kAsh(t):
    a = 0.32962
    b = 0.0014011
    c = -0.0000029069
    return a + b * t + c * t**2

def rhoPro(t):
    a = 1329.9
    b = -0.5184
    return  a + b * t

def rhoFat(t):
    a = 925.59
    b = -0.41757
    return a + b * t
    
def rhoCar(t):
    a = 1599.1
    b = -0.31046
    return a + b * t

def rhoFib(t):
    a = 1311.5
    b = -0.36589
    return a + b * t

def rhoAsh(t):
    a = 2423.8
    b = -0.28063
    return a + b * t

def cpPro(t):
    a = 2.0082
    b = 0.0012089
    c = -0.0000013129
    return (a + b * t + c * t**2) * 1000.


def cpFat(t):
    a = 1.9842
    b = 0.0014733
    c = -0.0000048008
    return (a + b * t + c * t**2) * 1000.

def cpCar(t):
    a = 1.5488
    b = 0.0019625
    c = -0.0000059399
    return (a + b * t + c * t**2) * 1000.

def cpFib(t):
    a = 1.8459
    b = 0.0018306
    c = -0.0000046509
    return (a + b * t + c * t**2) * 1000.

def cpAsh(t):
    a = 1.0926
    b = 0.0018896
    c = -0.0000036817
    return (a + b * t + c * t**2) * 1000.

def kw(t):
    return 0.57109 + 0.0017625 * t - 0.0000067036 * t**2


def kIcew(t):
    return 4.1559e-7 * t**3 + 9.7811e-5*t**2 - 7.0402e-3*t + 2.2170

def rhow(t):
    return 997.18 + 0.0031439 * t - 0.0037574 * t**2

def rhoIcew(t):
    return 916.89 - 0.13071 * t

def cpw(t):
    return (4.1289 - 0.000090864 *t + 0.0000054731 * t**2)*1000.
    
def cpIcew(t):
    return (2.0623 + 0.0060769 * t)*1000.

def tif(p, t0f):
    #P en MPa
    #t0f temperatura inicio congelacion alimento en C
    return t0f-0.072192*p - 0.000155*p**2

def tifk(p, t0fk):
    #t0fk temperatura inicio congelacion alimento en K
    return t0fk-0.072192*p - 0.000155*p**2


"""
def xi(p, tk, t0fk, xtw, xbw):
    #xtw: total water fraction
    #xbw: bounded water fraction
    #entra p en MPa
    #entra t en K
    if (tk > tifk(p, t0fk)):
        return 0.
    else:
        return (xtw - xbw)*(1.0-(tifk(p, t0fk) - tifk(p, 273.15))/(tk - tifk(p, 273.15)))
"""

def xi(tk, t0fk, xtw, xbw):
    if(tk > t0fk):
        return 0.
    else:
        #return (xtw - xbw) * (1.0 - t0fk / (tk + 1e-10))
        return 1.105 * xtw / (1.0 + 0.7138 / (np.log(t0fk - tk + 1.0)))
        
def rhof(xtw, xp, xfa, xc, xfi, xa, xi, t):
    #xtw: total water fraction
    #xi: ice fraction
    #rho liquid food
    #xufw: unfrozen water fraction
    xufw = xtw - xi
    summ = xufw/rhow(t) + xi/rhoIcew(t) + xp/rhoPro(t) + xfa/rhoFat(t)
    summ = summ + xc/rhoCar(t) + xfi/rhoFib(t) + xa/rhoAsh(t)
    return 1.0 / summ


def condf(xtw, xp, xfa, xc, xfi, xa, xi, t):
    xufw = xtw - xi
    rhoff = rhof(xtw, xp, xfa, xc, xfi, xa, xi, t)
    cond = rhoff*(xufw*kw(t)/rhow(t) + xi*kIcew(t)/rhoIcew(t) + xp*kPro(t)/rhoPro(t) + xfa*kFat(t)/rhoFat(t) +
                  xc*kCar(t)/rhoCar(t) + xfi*kFib(t)/rhoFib(t) + xa*kAsh(t)/rhoAsh(t))
    return cond
    
def cpf(xtw, xp, xfa, xc, xfi, xa, xi, t):
    xufw = xtw - xi
    return xufw*cpw(t) + xi*cpIcew(t) + xp*cpPro(t) + xfa*cpFat(t) + xc*cpCar(t) + xfi*cpFib(t) + xa*cpAsh(t)

def cpAp(xtw, xp, xfa, xc, xfi, xa, xi, t, tif, delt, hls):
    xufw = xtw - xi
    cps = cpf(xtw, xp, xfa, xc, xfi, xa, xi, tif - delt)
    cpl = cpf(xtw, xp, xfa, xc, xfi, xa, xi, tif + delt)
    if ((t > tif - delt) and (t < tif + delt)):
        return (cpl + cps)*0.5 + hls * 0.5 / delt
    else:
        return cpf(xtw, xp, xfa, xc, xfi, xa, xi, t)


