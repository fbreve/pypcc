# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:10:07 2020

@author: Fabricio Breve

based on Caio's demo:
[https://github.com/caiocarneloz/pycc/blob/master/demo.py](https://github.com/caiocarneloz/pycc/blob/master/demo.py)
            
Changes:
    1) Changed function name maskData() to hideLabels(), which seems more accurate.
    2) Changed Iris dataset to Wine dataset to match the Matlab implementation 
       example.
    3) Added the accuracy score.
    4) Renamed demo.py to example.py to match the Matlab implementation.
    5) Added data normalization step (StandardScaler)
    6) Raised the amount of iterations to get higher accuracy
    7) hideLabels() now use round() for rounding instead of int(), which rounds
       to the lowest integer
    
"""

LABEL_PERCENTAGE = 0.1
K_NN = 10
SEED = 0

import numpy as np
import time
from pcc import ParticleCompetitionAndCooperation
from sklearn.datasets import load_wine
#from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#FUNCTION FOR ENCODING STRING LABELS AND GENERATING "UNLABELED DATA"    
def hideLabels(true_labels, percentage, rng):
    labels = true_labels.copy()
    mask = np.ones(len(true_labels), dtype=bool)

    unique_labels = np.unique(true_labels)
    for enc, l in enumerate(unique_labels):
        idx = np.where(true_labels == l)[0]
        rng.shuffle(idx)
        n_hide = round(percentage * len(idx))
        mask[idx[:n_hide]] = False
        labels[labels == l] = enc

    labels[mask] = -1
    return labels.astype(int)


def run_pcc_example():
    #IMPORT DATASETS
    print("Loading the Wine dataset...")
    dataset = load_wine()  
    #dataset = load_digits()
    
    scaler = StandardScaler()
    data = scaler.fit_transform(dataset.data)
    labels = dataset.target
    
    #GENERATE UNLABELED DATA
    print(f"Randomly selecting {LABEL_PERCENTAGE*100:.0f}% of the elements to be presented to the algorithm with their labels...")
    rng = np.random.RandomState(SEED)
    masked_labels = hideLabels(labels, LABEL_PERCENTAGE, rng)
    
    #RUN THE MODEL
    print('Running the algorithm...')    
    start = time.time()
    model = ParticleCompetitionAndCooperation(impl="cython")
    model.build_graph(data, k_nn=K_NN)
    pred = np.array(model.fit_predict(masked_labels))
    end = time.time()

    elapsed = end - start

    # retorna tudo que pode ser útil para inspeção
    return model, pred, labels, masked_labels, elapsed


def main():
    model, pred, labels, masked_labels, elapsed = run_pcc_example()
    
    #SEPARATE PREDICTED SAMPLES
    hidden_labels = np.array(labels[masked_labels == -1]).astype(int)
    hidden_pred = pred[masked_labels == -1]
    
    #PRINT RESULTS
    print(f"\nAccuracy Score: {accuracy_score(hidden_labels, hidden_pred):.4f}")
    print(f"Execution Time: {elapsed:.4f}s")
    print("\nConfusion Matrix:\n", confusion_matrix(hidden_labels, hidden_pred))
    print("\nClassification Report:\n")
    print(classification_report(hidden_labels, hidden_pred))


if __name__ == "__main__":
    main()
