# -*- coding: utf-8 -*- 
'''
Created on 24/02/2014

@author: Jorge A. Zapata Guridi
'''
import numpy as np
import itertools

def AverageLinkage(H_orig,clusters,current_cluster,matrix_ind):
    new_row = []
    for k in clusters.keys():
        if(clusters[k] != None):
            matrix_ind.append(k)
            avg = 0
            for i in clusters[k]:
                for j in current_cluster:
                    avg += H_orig[i,j]
            avg = avg/(len(clusters[k])*len(current_cluster))
            new_row.append(avg)
    return new_row
            
def CompleteLinkage(H_orig,clusters,current_cluster,matrix_ind):
    new_row = []
    for k in clusters.keys():
        if(clusters[k] != None):
            matrix_ind.append(k)
            maxi = -1
            for i in clusters[k]:
                for j in current_cluster:
                    if H_orig[i,j] > maxi:
                        maxi = H_orig[i,j]
            new_row.append(maxi)
    return new_row
            
def SingleLinkage(H_orig,clusters,current_cluster,matrix_ind):
    new_row = []
    for k in clusters.keys():
        if(clusters[k] != None):
            matrix_ind.append(k)
            mini = 10
            for i in clusters[k]:
                for j in current_cluster:
                    if H_orig[i,j] < mini:
                        mini = H_orig[i,j]
            new_row.append(mini)
    return new_row

def Silhouette_Coefficient_Caldelli(H_orig,noises,clusters):
    si = []
    for ni in range(len(noises)):
        c = noises[ni]
        if len(clusters[c]) > 1:
            ai = [v for i,v in enumerate(H_orig[ni]) if i in set(clusters[c]) if i!=ni]
            ai = np.mean(ai)
        else:
            ai = 0
               
        #Determino la mínima distancia promedio al cluster mas cercano (más parecido)
        bi =  [np.mean(H_orig[[ni],clusters[i]]) for i in clusters if clusters[i] != None and i != c]
        bi = np.max(bi)
        if np.minimum(ai,bi) == 0:
            si.append((bi - ai)/np.maximum(ai,bi))
        else:
            si.append((bi - ai)/np.minimum(ai,bi))
    return np.mean(si)

def Silhouette_Coefficient(H_orig,noises,clusters):
    si = []
    for k in clusters.keys():
        if(clusters[k] != None):
            bi = []
            if len(clusters[k]) > 1:
                ai = [H_orig[p] for p in itertools.combinations(clusters[k],2)]
                ai = np.mean(ai)
            else:
                ai = 0
            
            for i in clusters:
                tmp = [] 
                if clusters[i] != None and i != k:
                    for nj in clusters[i]:
                        for ni in clusters[k]:
                            tmp.append(H_orig[ni,nj])
                    bi.append(np.mean(tmp))
            bi = np.max(bi)
            si.append(bi - ai)
            
    return np.mean(si)

def HierarchicalClustering(H,SIMILARITY_MSR='max'):
    H_orig = np.copy(H) #Guardo la matrix original y trabajo con la copia
    #Inicializo todos los clusters con 1 elemento
    #Z = []
    SCq = []
    clusters = {}
    noises = {}
    matrix_ind = []
    P = []
    for i in range(H.shape[0]):
        clusters[i] = [i]
        matrix_ind.append(i)
        noises[i] = i
        
    while len(H) != 1:
        #Busco el par de clusteres con mayor similitud
        M_size = H.shape[0]
        ind = np.triu_indices(M_size, 1)#Encima de diagonal principal
        if SIMILARITY_MSR == 'max':
            max_ind = np.argmax(H[ind])
        else:
            max_ind = np.argmin(H[ind])
        row, col = ind[0][max_ind], ind[1][max_ind]  
        m_row = matrix_ind[row]
        m_col = matrix_ind[col]
        
        #Elimino los clusters como estaban y creo el nuevo
        current_cluster = clusters[m_row] + clusters[m_col]
        #Z.append([m_row,m_col,H[row,col],len(current_cluster)])
        clusters[m_row] = None
        clusters[m_col] = None   #clusters.pop(m_col,None)
        
        #Reduzco la matriz
        H = np.delete(H,row,0)
        H = np.delete(H,col-1,0)
        H = np.delete(H,row,1)
        H = np.delete(H,col-1,1)
        
        #Actualizo la matriz
        matrix_ind = []
        new_row = AverageLinkage(H_orig,clusters,current_cluster,matrix_ind)
                
        matrix_ind.append(len(clusters)) #Agrego nuevo cluster formado a la matriz de indices
        noises.update(noises.fromkeys(current_cluster,len(clusters)))
        clusters[len(clusters)] = current_cluster #Nuevo cluster a la lista de clusters existentes
        H = np.vstack([H,new_row]) #Agrego fila
        new_row.append(1)
        new_col = np.array(new_row)
        H = np.hstack((H,new_col.reshape(-1,1))) #Agrego columna
        
        #Calcular el coeficiente Silhouette
        if len(H) != 1:
            SCq.append(Silhouette_Coefficient(H_orig, noises, clusters))
       
        #Salvo la partición
        tmp = clusters.copy()
        for b in tmp.keys():
            if tmp[b] == None:
                tmp.pop(b,None)
        P.append(tmp)
        
    p = SCq.index(min(SCq))
    return P[p]
