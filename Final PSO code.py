import random as rd
import statistics as st
import numpy as np
import pandas as pd
import copy as c
import sys

#importing and arranging data
av = pd.read_csv("C:/Rhimjhim/College/NTCC Sme 5/Final List of Companies.csv")
rc = av.shape[0]
cc = av.shape[1]
cl = av.columns.values.tolist()

#Markowitzfunction
def fit(pos):
    fv = 0.0
    rn = 0.0
    r = 0.0
    for i in range(0,cc):
        #returnfunction
        rn += pos[i]*(st.mean(av[cl[i]]))
        #riskfunction
        r += 2*(pos[i]**2)*(st.variance(av[cl[i]]))
        for j in range(i+1, cc):
            r += pos[i]*pos[j]*(st.covariance(av[cl[i]], av[cl[j]]))
    #fitnessfunction as a MarkowitzFunction
    fv = r - rn
    return fv

#Particleclass
class Particle:
    def __init__(self, fit, seed):
        self.rnd = rd.Random(seed)
        self.pos = [0.2 for i in range(cc)]
        self.velocity = [0.5 for i in range(cc)]
        self.bpp = [0.0 for i in range(cc)]

        for i in range(0, cc):
            self.pos[i] = self.rnd.random()
            self.velocity[i] = self.rnd.random()

        self.fit = fit(self.pos)

        self.bpp = c.copy(self.pos)
        self.bpf = self.fit

def pso(fit, max_iter, n):
    w = 0.7
    c1 = 1.7
    c2 = 1.9

    rnd = rd.Random(0.5)

    swarm = [Particle(fit, i) for i in range(n)]

    bsp = [0.2 for i in range(cc)]
    bsf = sys.float_info.max

    for i in range(n):
        if swarm[i].fit < bsf:
            bsf = swarm[i].fit
            bsp = c.copy(swarm[i].pos)
    
    Iter = 0
    while Iter < max_iter:
    
        for i in range(n):
            for k in range(cc):
                r1 = rnd.random()
                r2 = rnd.random()

                swarm[i].velocity[k] = ((w*swarm[i].velocity[k]) + (c1*r1*swarm[i].bpp[k] - swarm[i].pos[k]) + (c2*r2*(bsp[k]-swarm[i].pos[k])))
                
                if swarm[i].velocity[k]<0:
                    swarm[i].velocity[k] = 0
                elif swarm[i].velocity[k]>1:
                    swarm[i].velocity[k] = 1

            for k in range(cc):
                swarm[i].pos[k] += swarm[i].velocity[k]
            
            sum = np.sum(swarm[i].pos)
            for j in range(0, cc):
                swarm[i].pos[j] = round((swarm[i].pos[j]/sum), 8)

            swarm[i].fit = fit(swarm[i].pos)

            if swarm[i].fit < swarm[i].bpf:
                swarm[i].bpf = swarm[i].fit
                swarm[i].bpp = c.copy(swarm[i].pos)
            
            if swarm[i].fit < bsf:
                bsf = swarm[i].fit
                bsp = c.copy(swarm[i].pos)
        Iter+=1
    return bsp

fitness = fit
bp = pso(fitness, 50, 10)
print(bp)
fitnessval = fitness(bp)
print(fitnessval)