# -*- coding: utf-8 -*- 
import Functions
import numpy as np
#import time
import os
import re
import random
import scipy.io as sio
import matplotlib.pyplot as plt
#start_time = time.time()

SET = 'Exp17'
SPN_PATH = '/media/FreeAgent GoFlex Drive/PRNU/SPNs/Matlab/Final/'+SET+'/'
CENTROIDS_PATH = '../Centroids/'
CENTROID = 0

fig = plt.figure()
ax = fig.add_subplot(111)

files = os.listdir(SPN_PATH)
random.shuffle(files)
centroids = [f for f in os.listdir(CENTROIDS_PATH) if os.path.isfile(CENTROIDS_PATH+f)]

Centroid = np.load(CENTROIDS_PATH+centroids[0])
Modelos = {}
Colors = ['g','r','b','y','k','c','m']
Markers = ['o','v','+','*','h','x','s','d']
random.shuffle(Markers)
random.shuffle(Colors)


for f in range(len(files)):
    match = re.match(r'([a-z]*)(_[a-z0-9-]*)?',files[f])
    nombre = match.group()
    if nombre not in Modelos:
        Modelos[nombre] = {'x':[],'y':[]}
    
    tmp = sio.loadmat(SPN_PATH+files[f])
    tmp = tmp['Noisex']
    Modelos[nombre]['x'].append(f)
    Modelos[nombre]['y'].append(Functions.Correlation_fast(Centroid, tmp))

for i,cam in enumerate(Modelos):
    ax.scatter(Modelos[cam]['x'],Modelos[cam]['y'], s=30, c=Colors[i], marker=Markers[i])
    for j in Modelos[cam]['y']:
        print cam+','+str(i+1)+','+str(j)
    
plt.show()