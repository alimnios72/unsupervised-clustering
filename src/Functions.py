# -*- coding: utf-8 -*- 
'''
Created on 12/12/2013

@author: Jorge A. Zapata Guridi
'''

import numpy as np
import math


def SPN_Enhancer(SPN, alpha):
    ne = np.zeros(SPN.shape)
    
    for [i,j],val in np.ndenumerate(SPN):
        if val >= 0:
            ne[i,j] = np.exp(-0.5*((val**2)/alpha**2))
        else:
            ne[i,j] = -1 * np.exp(-0.5*((val**2)/alpha**2))
            
    return ne

def SPN_Enhancer_2(SPN, alpha):
    cond1 = (SPN < -alpha)
    cond2 = (SPN >= -alpha) & (SPN <= 0)
    cond3 = (SPN > 0) & (SPN <= alpha)
    cond4 = (SPN > alpha)
    SPN[cond1] = 0
    SPN[cond2] = -np.cos(SPN[cond2]*math.pi/(2*alpha))
    SPN[cond3] = np.cos(SPN[cond3]*math.pi/(2*alpha))
    SPN[cond4] = 0
    
    return SPN

def Correlation_slow(SPN1, SPN2):
    m1 = np.mean(SPN1)
    m2 = np.mean(SPN2)
    s1 = np.std(SPN1)
    s2 = np.std(SPN2)
    p = 0
    for [i,j],val in np.ndenumerate(SPN1):
        p += ((SPN1[i,j] - m1)*(SPN2[i,j] - m2))/(s1*s2)
    
    p = p/SPN1.size
    return p

def Correlation_fast(SPN1, SPN2):
    F = SPN1 - np.mean(SPN1)
    T = SPN2 - np.mean(SPN2)
    Fn = F / np.linalg.norm(F)
    Tn = T / np.linalg.norm(T)
    p = np.sum(Fn*Tn)
    
    return p


def CompareSets(set1,set2):
    m = set1.shape[0]
    n = set2.shape[0]
    M = np.empty((m,n))
    k = 0
    for i in range(m):
        for j in xrange(k,n):
            if i == j:
                M[i,j] = 1.0
            else:
                M[i,j] = Correlation_fast(set1[i],set2[j])
                M[j,i] = M[i,j]
        k += 1
    return M
      
def MaxSet(M1,M2,M3,M4):
    M = np.zeros(M1.shape) 
    for [i,j],val in np.ndenumerate(M1):
        M[i,j] =  np.array([val,M2[i,j],M3[i,j],M4[i,j]]).max()
    return M