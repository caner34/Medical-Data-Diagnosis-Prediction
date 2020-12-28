




import pandas as pd
import numpy as np
import os
import preprocessing_operative_services_prediction_by_diagnosis as preprocessed



X, y = preprocessed.get_data_splited()






from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
from sklearn.ensemble import RandomForestClassifier

def get_results_with_random_forest(X,y):
    
    X = np.array(X)
    y = np.array(y)
    
    stratified_folds = StratifiedKFold(n_splits=5, shuffle=True)
    
    for train_indices, test_indices in stratified_folds.split(X, y): 
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            clf = RandomForestClassifier(n_jobs=-2, class_weight='balanced')
            clf.fit(X_train, y_train)
            predictions = y.copy()
            predictions[test_indices] = clf.predict(X_test)
    
    
    print("cr_accuracy_score = ", accuracy_score(y, predictions))
    
    cr_precision_score = precision_score(y, predictions)
    cr_recall_score =  recall_score(y, predictions)
    cr_fbeta_score = fbeta_score(y, predictions, beta=2.0)

    print("cr_fbeta_score = ", cr_fbeta_score)
    print("cr_recall_score = ", cr_recall_score)
    print("cr_precision_score = ", cr_precision_score)





get_results_with_random_forest(X,y)






