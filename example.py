# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:10:07 2020

@author: Fabricio Breve

based on Caio's demo:
https://github.com/caiocarneloz/pycc/blob/master/demo.py
            
Changes:
    1) Changed function name maskData() to hideLabels(), which seems more accurate.
    2) Changed Iris dataset to Wine dataset to match the Matlab implementation 
       example.
    3) Added the accuracy score.
    4) Renamed demo.py to example.py to match the Matlab implementation.
    5) Added data normalization step (l2)
    6) Raised the amount of iterations to get higher accuracy
    7) hideLabels() now use round() for rounding instead of int(), which rounds
       to the lowest integer
    
"""

import numpy as np
import random
import time
from sklearn.datasets import load_wine
from pcc import ParticleCompetitionAndCooperation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA"
def hideLabels(true_labels, percentage):

    mask = np.ones((1,len(true_labels)),dtype=bool)[0]
    labels = true_labels.copy()
    
    for l, enc in zip(np.unique(true_labels),range(0,len(np.unique(true_labels)))):
        
        deck = np.argwhere(true_labels == l).flatten()
        random.shuffle(deck)
        
        mask[deck[:round(percentage * len(true_labels[true_labels == l]))]] = False

        labels[labels == l] = enc

    labels[mask] = -1
    
    return np.array(labels).astype(int)

#IMPORT DATASETS
print("Loading the Wine dataset...")
wine = load_wine()
data = normalize(wine.data,axis=0)
labels = wine.target

#GENERATE UNLABELED DATA
print("Randomly selecting 10% of the elements to be presented to the algorithm with their labels...")
masked_labels = hideLabels(labels, 0.1)

#RUN THE MODEL
print('Running the algorithm...')
start = time.time()
model = ParticleCompetitionAndCooperation()
model.build_graph(data,k_nn=10)
pred = np.array(model.fit_predict(masked_labels))
end = time.time()

#SEPARATE PREDICTED SAMPLES
hidden_labels = np.array(labels[masked_labels == -1]).astype(int)
hidden_pred = pred[masked_labels == -1]

#PRINT ACCURACY SCORE
print("Accuracy Score:", accuracy_score(hidden_labels,hidden_pred))

#PRINT TIME
print("Execution Time: " + "{0:.4f}".format(end-start) +'s')

#PRINT CONFUSION MATRIX
print("Confusion Matrix:\n", confusion_matrix(hidden_labels, hidden_pred))