# -*- coding: utf-8 -*- 
'''
Created on 04/12/2013

@author: Jorge A. Zapata Guridi
'''

import Functions
import Clustering as clr
import os
import numpy as np
import time
import scipy.io as sio
import re
from optparse import OptionParser


SET = 'Exp3'
FILE_TYPE = 'Matlab' #Software con que se generó el archivo de ruido
SPN_PATH = '/media/FreeAgent GoFlex Drive/PRNU/SPNs/'+FILE_TYPE+'/Final/'+SET+'/'
TEST_PATH = '../ImagesTest/'
CENTROIDS_PATH = '../Centroids/'
RESULTS_PATH = '../Results/'
MATRIX_PATH = '../Matrices/'+FILE_TYPE+'/Final/' #Ruta donde se guardan las matrices de similitud
ENHANCER = False #Aplico o no función de mejoramiento
EHC_FUN = 2    #Especifica función de mejoramiento [1] Chsunt T [2] Caldelli
SUFFIX = 'Enhanced '+str(EHC_FUN) if ENHANCER else 'No Enhanced'
MATRIX_FILE = MATRIX_PATH+SET+'('+SUFFIX+').npy'
WIDTH = 1024
HEIGHT = 1024
ALPHA1 = 7
ALPHA2 = 0.055
SIMILARITY_MSR = 'max' #Especifica la medida de relación entre clústeres

def Create_Similarity_Matrix(files):
    S = np.empty([len(files),WIDTH,HEIGHT])
    set_size = len(files)
    for i in range(set_size):
        if FILE_TYPE == 'Python':
            tmp = np.load(SPN_PATH+files[i])
        else:
            tmp = sio.loadmat(SPN_PATH+files[i])
            tmp = tmp['Noisex']
        if ENHANCER: #Aplico la función de mejoramiento
            if EHC_FUN == 1:
                tmp = Functions.SPN_Enhancer(tmp,ALPHA1)
            else:
                tmp = Functions.SPN_Enhancer_2(tmp,ALPHA2)    
        S[i] = tmp
    H = Functions.CompareSets(S, S)
    S = None
    #Guardo la matriz de correlaciones en un archivo para futuro uso
    np.save(MATRIX_FILE,H)
    return H

def EvaluateClusters(Clusters,files):
    Porcentajes = {}
    for c in Clusters.keys():
        Modelos = {}
        Total = 0
        for cam in Clusters[c]:
            match = re.match(r'([a-z]*)(_[a-z0-9-]*)?',files[cam])
            nombre = match.group()
            if nombre not in Modelos:
                Modelos[nombre] = {'num':1}
            else:
                Modelos[nombre]['num'] += 1
            Total += 1
        for i in Modelos.keys():
            Modelos[i]['pct'] =  100.0*Modelos[i]['num']/Total
        Porcentajes[c] = Modelos
    return Porcentajes

def GetClusterCentroids(Clusters,files,save=False):
    Centroids = {}
    for c in Clusters.keys():
        Spns = np.empty([len(Clusters[c]),WIDTH,HEIGHT])
        for i,ni in np.ndenumerate(Clusters[c]):
            if FILE_TYPE == 'Python':
                Spns[i] = np.load(SPN_PATH+files[ni])
            else:
                tmp = sio.loadmat(SPN_PATH+files[ni])
                Spns[i] = tmp['Noisex']
        Centroids[c] = np.mean(Spns,0)
        if save:
            np.save(CENTROIDS_PATH+str(c), Centroids[c])
    return Centroids

def SimpleClustering(Clusters,Centroids,files):
    TestFiles = os.listdir(TEST_PATH)
    for f in TestFiles:
        files.append(f)
        match = 0
        if FILE_TYPE == 'Python':
            tmp = np.load(TEST_PATH+f)
        else:
            tmp = sio.loadmat(TEST_PATH+f)
            tmp = tmp['Noisex']
        match = max(Centroids, key=lambda x: Functions.Correlation_fast(tmp, Centroids[x]))         
        Clusters[match].append(len(files)-1)
        
def SaveOutput(Clusters,Eval,files):
    out = open(RESULTS_PATH+SET+'.txt','w')
    for c in Clusters.keys():
        pct = ''
        for e in Eval[c].keys():
            pct += e+":"+str(Eval[c][e]['pct'])+"% "
        out.write("\n Cluster "+str(c)+" ["+pct+"]:\n")
        for f in Clusters[c]:
            out.write("\t"+files[f]+"\n")
    out.close()           
    
def main():
    parser = OptionParser()
    
    parser.add_option("-t", "--test", dest="test", default=False, action="store_true",
                  help="Test the given set with the centroids of clusters")
    parser.add_option("-s", "--save", dest="saveCentroids", default=False, action="store_true",
                  help="Save the centroids of each cluster in a file")
    parser.add_option("-f", "--force", dest="forceMatrix", default=False, action="store_true",
                  help="Creates similarity matrix even if it exists")
    
    (options, args) = parser.parse_args()
    
    start_time = time.time()    
    files = os.listdir(SPN_PATH)

    if os.path.isfile(MATRIX_FILE): #Si no existe la matrix de correlaciones la genero
        H = np.load(MATRIX_FILE)
    else:
        H = Create_Similarity_Matrix(files)
        
    print 'Trabajando con matriz.... '+MATRIX_FILE 
    print 'Tiempo de creación de matrix: ', time.time() - start_time
    
    Clusters = clr.HierarchicalClustering(H, SIMILARITY_MSR)
    
    if options.test:
        Centroids = GetClusterCentroids(Clusters,files,options.saveCentroids)
        SimpleClustering(Clusters,Centroids,files)
    
    Eval = EvaluateClusters(Clusters,files)
    
    SaveOutput(Clusters,Eval,files)
    print "Numero total de clusters/camaras = "+str(len(Clusters))
    print time.time() - start_time, "seconds"        

if __name__ == '__main__':
    main()
