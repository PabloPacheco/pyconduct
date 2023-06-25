import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import coo_matrix
import scipy.sparse.linalg
from numba import jit
from scipy.interpolate import griddata

class PyConduct():
    def __init__(self):
        # Mesh
        self.lx = 1.
        self.ly = 1.
        self.ncvx = 5
        self.ncvy = 5
        # Plot
        self.name = "contour"
        self.ndec = 2
        self.cmapStyle = "jet"
        # Monitors
        self.monitors = np.array([[0,0], [self.lx * 0.5, self.ly * 0.5], [self.lx, self.ly]])
        self.monNames = np.array(["node1", "node2", "node3"])
        self.reportMon = 10
        # Control
        self.tolnl = 1e-6
        self.alertnl = 100
        self.reportBlackW = 50
        self.reportContour = 1000
        self.timeEnd = 1.
        self.gammaFunction = gammaPYC
        self.lambFunction = lambPYC
        self.unsteadySourcesFunction = unsteadySourcesPYC
        self.sourcesVolFunction = sourcesVolPYC
        self.sourcesBoundFunction = sourcesBoundPYC

        
    def set_default(self):
        self.time = 0.
        self.dt = 1.
        ncvt = self.ncvx * self.ncvy
        nft = self.ncvx * self.ncvy * 2 + self.ncvx + self.ncvy
        nfi = nft - 2 * (self.ncvx + self.ncvy)
        nfb = nft - nfi
        nct = self.ncvx * self.ncvy * 4
        nci = nct - 2 * self.ncvx - 2 * self.ncvy
        ncb = nct - nci
        nvx = self.ncvx + 1
        nvy = self.ncvy + 1
        nctt = nct + self.ncvx * self.ncvy
        ncit = nci + self.ncvx * self.ncvy
        self.phi = np.ones(ncvt)
        self.phi1 = np.ones(ncvt)
        self.phi2 = np.ones(ncvt)
        self.gamma = np.ones(ncvt)
        self.lamb = np.ones(ncvt)
        self.phif = np.ones(nfb)        
        self.ndist = np.zeros(nctt)
        self.ndistf = np.zeros((nctt, 2))       
        self.sc = np.zeros(ncvt)
        self.sp = np.zeros(ncvt)
        self.coef = np.zeros(nctt)
        self.ap = np.zeros(ncvt)
        self.b = np.zeros(ncvt)
        self.apt = np.zeros(ncvt)
        self.act = np.zeros(ncvt)
        # solver
        # 1: Direct, 2: CG, 3: GMRES, 4: LGMRES, 5: MINRES, 6: QMR
        self.sol = 2
        self.tol = 1e-6
        self.cellC = np.zeros((nctt, 4), dtype = int)
        self.tolNl = 1e-6
        self.alertNl = 100
        self.writeContour = 250
        self.reportBlackW = 50
        self.reportContour = 1000
        self.indCI = np.zeros(ncit, dtype = int)
        self.indCB = np.zeros(ncb, dtype = int)
        self.indCN = np.zeros(nci, dtype = int)
        self.indCII = np.zeros(self.ncvx * self.ncvy, dtype = int)
        self.diffc = np.zeros(nci, dtype = float)
        self.vvrr = np.zeros(len(self.cellC[:,0]))
        self.nmon = len(self.monitors[:,0])
        self.nodev = np.zeros(self.nmon, dtype=int)
        self.vertexC = np.zeros((self.ncvx * self.ncvy, 4), dtype = int)
        self.vertexID = np.linspace(0, nvx * nvy - 1, nvx * nvy , dtype = int)
        self.xv = np.zeros(nvx * nvy, dtype = float)
        self.yv = np.zeros(nvx * nvy, dtype = float)
        self.x1d = np.linspace(0., self.lx, nvx)
        self.y1d = np.linspace(0., self.ly, nvy)
        self.faceV = np.zeros((nft, 2), dtype = int)
        self.vertexC = np.zeros((self.ncvx * self.ncvy, 4), dtype = int)
        self.vertexID = np.linspace(0, nvx * nvy - 1, nvx * nvy , dtype = int)
        self.xf = np.zeros(nft, dtype = float)
        self.yf = np.zeros(nft, dtype = float)
        self.af = np.zeros(nft)
        self.nf = np.zeros((len(self.indCN), 2))
        self.cvol = np.zeros(ncvt)
        self.bcnS = np.zeros(self.ncvx, dtype = int)
        self.bcnN = np.zeros(self.ncvx, dtype = int)
        self.bcnE = np.zeros(self.ncvy, dtype = int)
        self.bcnW = np.zeros(self.ncvy, dtype = int)
        self.sources = np.zeros(ncvt, dtype = int)
        return
    
    def meshing(self):
        self.xv, self.yv = ugridVertex(self.ncvx, self.ncvy, self.lx, self.ly, self.xv, self.yv, self.x1d, self.y1d)
        self.faceV = faceVertexC(self.ncvx, self.ncvy, self.faceV)
        self.vertexC, self.vertexID = vertex(self.ncvx, self.ncvy, self.vertexC, self.vertexID)
        self.cellC = cellConect(self.ncvx, self.ncvy, self.cellC)
        self.indCI, self.indCB, self.indCN, self.indCII = \
        indexC(self.ncvx, self.ncvy, self.cellC, self.indCI, self.indCB, self.indCN, self.indCII)
        self.xc, self.yc = cellCenter2(self.vertexC, self.xv, self.yv, self.ncvx, self.ncvy)
        self.xf, self.yf = faceCenter(self.faceV, self.xv, self.yv, self.ncvx, self.ncvy, self.xf, self.yf)
        self.af = Af(self.ncvx, self.ncvy, self.faceV, self.xf, self.yf, self.xv, self.yv, self.af)
        self.nf = faceNormal(self.indCN, self.cellC, self.xc, self.yc, self.nf)
        self.cvol = volume(self.vertexC, self.xv, self.yv, self.ncvx, self.ncvy, self.cvol)
        self.bcnS, self.bcnE, self.bcnN, self.bcnW, self.sources = \
        boundaryIndex(self.ncvx, self.ncvy, self.cellC, self.indCB, \
                      self.bcnS, self.bcnN, self.bcnE, self.bcnW, self.sources) 
        self.ndist, self.ndistf = nodalDist2(self.ncvx, self.ncvy, \
                                             self.cellC, self.xc, self.yc, self.xf, self.yf, self.ndist, self.ndistf)
        return 
    
    def run_simulation(self):

        self.I = self.cellC[self.indCI,0]
        self.J = self.cellC[self.indCI,1]

        nodesM = findNodes(self.monitors, self.xv, self.yv, self.xc, self.yc, self.vertexC, self.nodev, self.nmon)
        archMonitors = open("monitors.csv", "w")

        self.phi1[:] = self.phi[:]
        self.phi2[:] = self.phi[:]

        timer = 0.
        titer = 0

        while timer < self.timeEnd:

            errNl = 1.
            countIterNl = 0

            while errNl >= self.tolNl:
                phi0Nl = self.phi
                
                self.gamma = calcGamma(self.phi, self.gamma, self.xc, self.yc, self.gammaFunction)
                self.lamb = calcLamb(self.phi, self.lamb, self.xc, self.yc, self.lambFunction)
                    
                self.sc, self.sp = \
                calcSourcesB(self.indCB, self.cellC, self.sc, self.sp, self.ndist, \
                             self.cvol, self.af, self.gamma, self.lamb, self.xf, self.yf, timer, self.sourcesBoundFunction) 
                self.sc, self.sp = calcSourcesV(self.sc, self.sp, self.cvol, self.phi, self.xc, self.yc, timer, self.sourcesVolFunction) 
                self.act, self.apt = \
                unsteadyScSp(self.phi, self.phi1, self.phi2, \
                             self.lamb, self.dt, self.cvol, titer, self.act, self.apt, self.unsteadySourcesFunction)

                self.diffc = diffCoef(self.ndistf, self.gamma, self.indCN, self.cellC, self.af, self.diffc)
                self.ap = calcAp(self.indCN, self.cellC, self.sp, self.diffc, self.ap, self.apt)
                self.b = calcbNP(self.sc, self.act, self.b)
                self.vvrr = VAL(self.cellC, self.indCN, self.indCII, self.diffc, self.ap, self.vvrr)
                self.A = matrixAssemble(self.indCI, self.vvrr, self.cellC, self.ncvx, self.ncvy, self.I, self.J)
                self.phi = solvers(self.phi, self.b, self.A, self.tol, self.sol)
                self.phif = solverBound2(self.cellC, self.indCB, self.ndist, self.sc, self.sp, self.phi, self.gamma, self.phif, self.ncvx, self.ncvy)
                self.vvrr, self.ap, self.sc, self.sp = resetTime(self.vvrr, self.ap, self.sc, self.sp)
                errNl = np.mean((phi0Nl - self.phi)**2)
                phi0Nl = self.phi
                countIterNl += 1
                if countIterNl > self.alertNl:
                    print(f"more than {self.alertNl} iterations have been executed due to non-linearity")

                #Postprocessing
            if titer % self.writeContour == 0:
                writeFile(self.cellC, self.indCB, self.ncvx, self.ncvy, self.xc, self.yc, self.xf, self.yf, self.phi, self.phif, self.name, self.ndec , timer)
            if titer % self.reportContour == 0:
                writeFile(self.cellC, self.indCB, self.ncvx, self.ncvy, self.xc, self.yc, self.xf, self.yf, self.phi, self.phif, self.name, self.ndec , timer)
                contour2(self.lx, self.ly, self.ncvx, self.ncvy, self.name, timer, self.ndec, self.cmapStyle)

            generateMonitors(archMonitors, self.phi, titer, self.monNames, nodesM, self.reportMon, timer)  

            #update phi
            self.phi2[:] = self.phi1[:]
            self.phi1[:] = self.phi[:]

            #next time step
            timer += self.dt
            titer += 1

            if titer % self.reportBlackW == 0:
                print("time = " + str(timer) + "  iter_t = " +  str(titer) + "  nl_iter = " \
                      + str(countIterNl - 1) + "  errNl = " + str(errNl))

        archMonitors.close()
        print("calculation complete")        


@jit
def calcGamma(phi, gamma, xc, yc, function):
    for i in range(len(phi)):
        gamma[i] = function(phi[i], xc[i], yc[i])
    return gamma

@jit
def calcLamb(phi, lamb, xc, yc, function):
    for i in range(len(phi)):
        lamb[i] = function(phi[i], xc[i], yc[i])
    return lamb

@jit
def calcSourcesB(indCB, cellC, sc, sp, ndist, cvol, af, gamma, lamb, xf, yf, timer, function):
    for i in range(len(indCB)):
        ind = indCB[i]       
        c1 = cellC[ind,0]
        idb = cellC[ind,2]
        idf = cellC[ind,3]        
        sci, spi = function(idb, c1, idf, ind, ndist, cvol, af, gamma, lamb, xf, yf, timer)
        sc[c1] += sci
        sp[c1] += spi
    return sc, sp


@jit
def calcSourcesV(sc, sp, cvol, phi, xc, yc, timer, function):
    for i in range(len(sc)):
        sci, spi = function(i, cvol[i], phi, xc, yc, timer)
        sc[i] += sci
        sp[i] += spi
    return sc, sp


@jit      
def unsteadyScSp(phi, phi1, phi2, lamb, dt, cvol, titer, act, apt, function):
    for i in range(len(phi)):
        apt0, act0 = function(phi, phi1, phi2, lamb, dt, cvol, i, titer)
        act[i] = act0
        apt[i] = apt0
    return act, apt


# 1: Direct, 2: CG, 3: GMRES, 4: LGMRES, 5: MINRES, 6: QMR
def solvers(phi, b, A, tol, sol):
    if (sol == 2):
        phi, success = scipy.sparse.linalg.cg(A, b, tol = tol, x0 = phi)
    elif (sol == 3):
        phi, success = scipy.sparse.linalg.gmres(A, b, tol = tol, x0 = phi)
    elif (sol == 4):
        phi, success = scipy.sparse.linalg.lgmres(A, b, tol = tol, x0 = phi)
    elif (sol == 5):
        phi, success = scipy.sparse.linalg.minres(A, b, tol = tol, x0 = phi)
    elif (sol == 6):
        phi, success = scipy.sparse.linalg.qmr(A, b, tol = tol, x0 = phi)
    else:
        phi = scipy.sparse.linalg.spsolve(A, b)
    return phi


@jit
def ugridVertex(ncvx, ncvy, lx, ly, xv, yv, x1d, y1d):
    nvx = ncvx + 1
    nvy = ncvy + 1
    count = 0
    for j in range(nvy):
        for i in range(nvx):
            xv[count] = x1d[i] 
            count += 1  
    count = 0
    for j in range(nvy):
        for i in range(nvx):
            yv[count] = y1d[j] 
            count += 1       
    return xv, yv


@jit
def faceVertexC(ncvx, ncvy, faceV):
    nft = ncvx * ncvy * 2 + ncvx + ncvy
    nvx = ncvx + 1
    nvy = ncvy + 1
    # Caras horizontales
    start = 0
    add = 2 * ncvx + 1
    startL = 0
    addL = nvx
    startR = 1
    addR = nvx

    for j in range(nvy):
        for i in range(ncvx):
            faceV[start + i, 0] = startL + i
            faceV[start + i, 1] = startR + i
        start += add
        startL += addL
        startR += addR

    #caras verticales
    start = ncvx
    add = 2 * ncvx + 1
    startD = 0
    addD = nvx
    startU = nvx
    addU = nvx
    for j in range(ncvy):
        for i in range(ncvx + 1):
            faceV[start + i, 0] = startD + i
            faceV[start + i, 1] = startU + i
        start += add
        startD += addD
        startU += addU
    return faceV 


@jit
def vertex(ncvx, ncvy, vertexC, vertexID):
    nvx = ncvx + 1
    nvy = ncvy + 1
    ncvt = ncvx * ncvy
    suma = 0
    count = 0
    for i in range(nvy - 1):
        for i in range(nvx - 1):
            vertexC[count, 0] = i + suma
            vertexC[count, 1] = 1 + i + suma
            vertexC[count, 2] = nvx + i + 1 + suma
            vertexC[count, 3] = nvx + i + suma
            count += 1
        suma += nvx
    return vertexC, vertexID


@jit
def indexC(ncvx, ncvy, cellC, indCI, indCB, indCN, indCII): 
    nct = ncvx * ncvy * 4
    nci = nct - 2 * ncvx - 2 * ncvy
    ncb = nct - nci
    nctt = nct + ncvx * ncvy
    ncit = nci + ncvx * ncvy
    
    countb = 0
    counti = 0
    countii = 0
    countn = 0
    
    for i in range(nctt):
        if (cellC[i, 1] < 0):
            indCB[countb] = i
            countb += 1
        else:
            indCI[counti] = i
            counti += 1
        if (cellC[i,1] >= 0 and cellC[i,0] != cellC[i,1]):
            indCN[countn] = i
            countn += 1
        if (cellC[i,0] == cellC[i,1]):
            indCII[countii] = i
            countii += 1
    return indCI, indCB, indCN, indCII




def cellCenter2(vertexC, xv, yv, ncvx, ncvy):
    ncvt = ncvx * ncvy
    xc = np.zeros(ncvt)
    yc = np.zeros(ncvt)
    xc[:] = (xv[vertexC[:,0]] + xv[vertexC[:,1]] + xv[vertexC[:,2]] + xv[vertexC[:,3]]) / 4.0
    yc[:] = (yv[vertexC[:,0]] + yv[vertexC[:,1]] + yv[vertexC[:,2]] + yv[vertexC[:,3]]) / 4.0
    return xc, yc

@jit
def faceCenter(faceV, xv, yv, ncvx, ncvy, xf, yf):
    nft = ncvx * ncvy * 2 + ncvx + ncvy
    for i in range(nft):
        xf[i] = 0.5 * (xv[faceV[i,0]] + xv[faceV[i,1]])
        yf[i] = 0.5 * (yv[faceV[i,0]] + yv[faceV[i,1]])
    return xf, yf

@jit
def Af(ncvx, ncvy, faceV, xf, yf, xv, yv, af):
    nft = ncvx * ncvy * 2 + ncvx + ncvy
    for i in range(nft):
        idv1 = faceV[i,0]
        idv2 = faceV[i,1]
        af[i] = np.sqrt((xv[idv1] - xv[idv2])**2 + (yv[idv1] - yv[idv2])**2)
    return af

@jit
def faceNormal(indCN, cellC, xc, yc, nf):
    for i in range(len(indCN)):
        ind = indCN[i]
        c1 = cellC[ind, 0]
        c2 = cellC[ind, 1]
        x1 = xc[c1]
        y1 = yc[c1]
        x2 = xc[c2]
        y2 = yc[c2]
        dx = x2 - x1
        dy = y2 - y1
        da = np.sqrt(dx**2 + dy**2)
        nf[i,:] = np.array([dx / da, dy / da]) * -1.0
    return nf

@jit
def volume(vertexC, xv, yv, ncvx, ncvy, cvol):
    ncvt = ncvx * ncvy
    for i in range(ncvt):
        x0 = xv[vertexC[i,1]]
        x1 = xv[vertexC[i,0]]
        y0 = yv[vertexC[i,1]]
        y1 = yv[vertexC[i,2]]
        base = np.abs(x0 - x1)
        altura = np.abs(y0 - y1)
        cvol[i] = base * altura
    return cvol

@jit
def boundaryIndex(ncvx, ncvy, cellC, indCB, bcnS, bcnN, bcnE, bcnW, sources):
    #numero de conexiones totales
    nct = ncvx * ncvy * 4
    #numero de conexiones internas
    nci = nct - 2 * ncvx - 2 * ncvy
    #numero de conexiones de borde
    ncb = nct - nci
    #numero de conexiones totales
    #considerando conexion consigo mismo
    nctt = nct + ncvx * ncvy
    ncit = nci + ncvx * ncvy
    ncvt = ncvx * ncvy

    countS = 0
    countN = 0
    countE = 0
    countW = 0
    countC = 0

    for i in indCB:
        if(cellC[i,2] == 1):
            bcnS[countS] = i
            countS += 1
        elif(cellC[i,2] == 2):
            bcnE[countE] = i
            countE += 1
        elif(cellC[i,2] == 3):
            bcnN[countN] = i
            countN += 1
        else:
            bcnW[countW] = i
            countW += 1

    for i in range(nctt):
        if(cellC[i,0] == cellC[i,1]):
            sources[countC] = i
            countC += 1
    return bcnS, bcnE, bcnN, bcnW, sources


@jit
def nodalDist2(ncvx, ncvy, cellC, xc, yc, xf, yf, ndist, ndistf):
    nct = ncvx * ncvy * 4
    nctt = nct + ncvx * ncvy
    for i in range(nctt):
        if ( cellC[i,1] >= 0):
            #conexion CV
            c1 = cellC[i,0]
            c2 = cellC[i,1]

            if(c1 == c2):
                ndistf[i,0] = 0.
                ndistf[i,1] = 0.
                ndist[i] = 0.
            else:
                cf = cellC[i,3]
                x1 = xc[c1]
                x2 = xc[c2]
                y1 = yc[c1]
                y2 = yc[c2]
                xcf = xf[cf]
                ycf = yf[cf]
                ndistf[i,0] = np.sqrt((x1 - xcf)**2 + (y1 - ycf)**2)
                ndistf[i,1] = np.sqrt((x2 - xcf)**2 + (y2 - ycf)**2)
                ndist[i] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        else:
            #conexion de borde
            idc = cellC[i,0]
            idf = cellC[i,3]
            ndistf[i,0] = np.sqrt((xc[idc] - xf[idf])**2 + (yc[idc] - yf[idf])**2)
            ndistf[i,1] = 0.
            ndist[i] = np.sqrt((xc[idc] - xf[idf])**2 + (yc[idc] - yf[idf])**2)
    return ndist, ndistf



@jit
def diffCoef(ndistf, gamma, indCN, cellC, af, diffc):
    for i in range(len(indCN)):
        ind = indCN[i]
        c1 = cellC[ind,0]
        c2 = cellC[ind,1]
        indf = cellC[ind,3]
        diffc[i] = (ndistf[ind,0] / (gamma[c1] + 1.0e-20) + ndistf[ind,1] / (gamma[c2] + 1.0e-20))**(-1) * af[indf]
    return diffc   


@jit
def calcAp(indCN, cellC, sp, diffc, ap, apt):
    for i in range(len(indCN)):
        ind = indCN[i]
        c1 = cellC[ind, 0]
        ap[c1] += diffc[i]
    for i in range(len(sp)):
        ap[i] -= (sp[i] + apt[i])
    return ap


def calcbNP(sc, act, b):
    b[:] = sc[:] + act[:]
    return b

@jit
def VAL(cellC, indCN, indCII, anb, ap, vvrr):
    for i in range(len(indCII)):
        ind = indCII[i]
        vvrr[ind] = ap[i]
    for i in range(len(indCN)):
        ind = indCN[i]
        vvrr[ind] = - anb[i]
    return vvrr

def matrixAssemble(indCI, vvrr, cellC, ncvx, ncvy, I, J):
    ncvt = ncvx * ncvy
    VR = vvrr[indCI]
    A = coo_matrix((VR, (I, J)), shape=(ncvt, ncvt))
    A = A.tocsr()
    return A

@jit
def solverBound2(cellC, indCB, ndist, sc, sp, phi, gamma, phif, ncvx, ncvy):
    ncvt = ncvx * ncvy
    nft = ncvx * ncvy * 2 + ncvx + ncvy
    nfi = nft - 2 * (ncvx + ncvy)
    nfb = nft - nfi
    for i in range(nfb):
        indf = indCB[i]
        c1 = cellC[indf, 0]
        idf = cellC[indf, 3]
        bound = cellC[indf, 2]
        dx = ndist[indf]
        phif[i] = (sc[c1] + gamma[c1] / dx * phi[c1]) / (gamma[c1] / dx - sp[c1])
    return phif

def writeFile(cellC, indCB, ncvx, ncvy, xc, yc, xf, yf, phi, phif, name, ndec , timer):
    #writeFile
    nvx = ncvx + 1
    nvy = ncvy + 1
    ncvt = ncvx * ncvy
    nft = ncvx * ncvy * 2 + ncvx + ncvy
    nfi = nft - 2 * (ncvx + ncvy)
    nfb = nft - nfi
    archiv = open(name + "_" + str(np.round(timer, ndec)) + ".dat","w")
    archiv.write('TITLE = "PHI"\n')
    archiv.write('VARIABLES="X","Y","PHI"\n')
    archiv.write('ZONE T="VAR" I='+str(nvx)+' J='+str(nvy)+"\n")

    for i in range(ncvt):
        archiv.write(str(xc[i]) + " " + str(yc[i]) + " " + str(phi[i]) + "\n")

    for i in range(nfb):
        face = cellC[indCB[i],3]
        archiv.write(str(xf[face]) + " " + str(yf[face]) + " " + str(phif[i]) + "\n")
    archiv.close()
    return 

def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi, method="linear")
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

def contour2(lx, ly, ncvx, ncvy, name, time, ndec, cmapStyle):
    name = name + "_" + str(np.round(time, ndec)) + ".dat"
    data = np.loadtxt(name, skiprows=3)    
    gx, gy = np.mgrid[0:lx:100j,0:ly:100j]
    grid_Z = griddata(data[:,0:2], data[:,2], (gx, gy), method='linear')
    plt.imshow(grid_Z.T, extent=(0,lx,0,ly), origin='lower', cmap = cmapStyle)
    plt.colorbar()
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    #plt.grid(None)
    plt.grid(False)
    plt.show()
    return 

def generateMonitors(archMonitors, phi, titer, monNames, nodesM, reportMon, timer):
    if titer == 0:
        archMonitors.write("t,")
        for i  in range(len(monNames)):
            archMonitors.write(monNames[i])
            if i == (len(monNames) - 1):
                archMonitors.write("\n")
            else:
                archMonitors.write(",")
    if titer % reportMon == 0:
        archMonitors.write(str(timer) + ",")
        for i in range(len(monNames)):
            archMonitors.write(str(phi[nodesM[i]]))
            if i == (len(monNames) - 1):
                archMonitors.write("\n")
            else:
                archMonitors.write(",")
    return 

@jit
def findNodes(monitors, xv, yv, xc, yc, vertexC, nodev, nmon):
    count = 0
    for j in range(nmon):
        for i in range(len(vertexC[:,0])):
            dx = xv[vertexC[i,1]] - xv[vertexC[i,0]]
            dy = yv[vertexC[i,1]] - yv[vertexC[i,2]]
            diag = np.sqrt((dx * 0.5)**2 + (dy * 0.5)**2)
            diag = diag + diag * 0.02
            xx = xc[i]
            yy = yc[i]
            dist = np.sqrt((xc[i] - monitors[j,0])**2 + (yc[i] - monitors[j,1])**2)
            if dist <= diag:
                nodev[count] = i
                count += 1
                break
    return nodev

def resetTime(vvrr, ap, sc, sp):
    vvrr[:] = 0.
    ap[:] = 0.
    sc[:] = 0.
    sp[:] = 0.
    return vvrr, ap, sc, sp

@jit
def cellConect(ncvx, ncvy, cellC):
    maxi = ncvx - 1
    maxj = ncvy - 1

    nct = ncvx * ncvy * 4
    nci = nct - 2 * ncvx - 2 * ncvy
    ncb = nct - nci
    nctt = nct + ncvx * ncvy
    ncit = nci + ncvx * ncvy

    count = 0
    countC = 0
    startS = 0
    startE = ncvx + 1
    startN = 2 * ncvx + 1
    startW = ncvx
    add = ncvx * 2 + 1

    for j in range(ncvy):
        for i in range(ncvx):
            if (i == 0 and j == 0):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 1
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC
                cellC[count, 1] = 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 4
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i > 0 and i < maxi and j == 0):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 1
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = countC + 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = countC + ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i == maxi and j == 0):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 1
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = -1
                cellC[count, 2] = 2
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = countC + ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i == 0 and j > 0 and j < maxj):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = countC + 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = countC + ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 4
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i == maxi and j > 0 and j < maxj):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = -1
                cellC[count, 2] = 2
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = countC + ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i== 0 and j == maxj):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = countC + 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 3
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 4
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i > 0 and i < maxi and j == maxj):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = countC + 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 3
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            elif(i == maxi and j == maxj):
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = -1
                cellC[count, 2] = 2
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = -1
                cellC[count, 2] = 3
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            else:
                #cara S
                cellC[count, 0] = countC
                cellC[count, 1] = countC - ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startS + i
                count += 1
                #cara E
                cellC[count, 0] = countC 
                cellC[count, 1] = countC + 1
                cellC[count, 2] = 0
                cellC[count, 3] = startE + i
                count += 1
                #cara N
                cellC[count, 0] = countC
                cellC[count, 1] = countC + ncvx
                cellC[count, 2] = 0
                cellC[count, 3] = startN + i
                count += 1
                #cara W
                cellC[count, 0] = countC
                cellC[count, 1] = countC - 1
                cellC[count, 2] = 0
                cellC[count, 3] = startW + i
                count += 1
                #Consigo mismo 
                cellC[count, 0] = countC
                cellC[count, 1] = countC
                cellC[count, 2] = 0
                cellC[count, 3] = -1
                count += 1

            countC += 1
        startS += add
        startE += add
        startN += add
        startW += add
    return cellC

@jit
def gammaPYC(t, xc, yc):
    gam = 1.
    return gam

@jit
def lambPYC(t, xc, yc):
    lamb = 1.
    return lamb

@jit
def sourcesBoundPYC(b, c, f, i, ndist, cvol, af, gamma, lamb, xf, yf, timer):
    #- 3: N
    #- 1: S
    #- 2: E
    #- 4: W
    gam = gamma[c]
    dx = ndist[i]
    vol = cvol[c]
    area = af[f]
    tb1 = 1.
    tb0 = 0.
    
    if(b == 1):
        scb = tb0 * gam / dx * area
        spb = - gam / dx * area
    elif(b == 2):
        scb = 0.
        spb = 0.
    elif(b == 3):
        scb = tb1 * gam / dx * area
        spb = - gam / dx * area
    elif(b == 4):
        scb = tb0 * gam / dx * area
        spb = - gam / dx * area
    else:
        scb = 0.
        spb = 0.
    return scb, spb

@jit
def sourcesVolPYC(c, vol, phi, xc, yc, timer):
    scv = 0.
    spv = 0.
    return scv, spv

@jit              
def unsteadySourcesPYC(phi, phi1, phi2, lamb, dt, cvol, c, titer):
    vol = cvol[c]
    #Euler
    ap0 = lamb[c] / dt * vol
    apts = -ap0 
    acts = ap0 * phi1[c]
    return apts, acts