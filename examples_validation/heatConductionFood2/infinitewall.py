import numpy as np

class InfiniteWall:
    def __init__(self, data):
        L, T0, Tf, h, k, rho, cp = data
        Bi = h * L / k
        lamb = np.linspace(1e-7, np.pi * 20, 1000000)
        fleft = np.ones(len(lamb)) * Bi
        fright = lamb * np.tan(lamb)
        idx = np.argwhere(np.diff(np.sign(fleft - fright))).flatten()
        intersect = lamb[idx]
        ind = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
        lambv = intersect[ind]
        An = np.zeros_like(lambv)
        An[:] = 4. * np.sin(lambv[:]) / (2. * lambv[:] + np.sin(2. * lambv[:]))
        self.lambv = lambv
        self.T0, self.Tf = T0, Tf
        self.An = An
        self.Bi = Bi
        self.L = L
        self.alpha = k / (rho * cp)
        
    def calcTemp(self, t, x):
        suma = 0.
        tau = self.alpha * t / (self.L**2)
        for i in range(len(self.An)):
            suma += self.An[i] * np.exp(-self.lambv[i]**2 * tau) * np.cos(self.lambv[i] * x / self.L)
        theta = suma
        T = theta * (self.T0 - self.Tf) + self.Tf
        return T
    
    def calcTheta(self, t, x):
        suma = 0.
        tau = self.alpha * t / (self.L**2)
        for i in range(len(self.An)):
            suma += self.An[i] * np.exp(-self.lambv[i]**2 * tau) * np.cos(self.lambv[i] * x / self.L)
        theta = suma
        return theta