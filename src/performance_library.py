# performance_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used by the depression prediction model. It includes functions that calculate performance metrics of
# a model represented as the list of predictions "y_pred" compared against the true targets "y_true".

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def specificity_score(y_true, y_pred):
    """
    It calculates a specificity score of a model comparing a list of predictions against a list of true values of the target. 
    
    Inputs:
    - y_true: list of true values of the target variable
    - y_pred: list of predicted values of the target variable
    
    Outputs:
    - a specificity score of the model used to produce the predictions in 'list y_pred'
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn + fp)


def precision_score(y_true, y_pred):
    """
    It calculates a precision score of a model comparing a list of predictions against a list of true values of the target. 
    
    Inputs:
    - y_true: list of true values of the target variable
    - y_pred: list of predicted values of the target variable
    
    Outputs:
    - a precision score of the model used to produce the predictions in 'list y_pred'
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp + fp)


def recall_score(y_true, y_pred):
    """
    It calculates a recall score of a model comparing a list of predictions against a list of true values of the target. 
    
    Inputs:
    - y_true: list of true values of the target variable
    - y_pred: list of predicted values of the target variable
    
    Outputs:
    - a recall score of the model used to produce the predictions in 'list y_pred'
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp + fn)


def accuracy_score(y_true, y_pred):
    """
     It calculates an accuracy score of a model comparing a list of predictions against a list of true values of the target. 
    
    Inputs:
    - y_true: list of true values of the target variable
    - y_pred: list of predicted values of the target variable
    
    Outputs:
    - an accuracy score of the model used to produce the predictions in 'list y_pred'
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp + tn)/(tp + tn + fp + fn)


