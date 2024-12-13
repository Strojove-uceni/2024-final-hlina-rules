from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import time
import torch
import numpy as np

def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    cf = confusion_matrix(labels, predictions)
    metrics_dict = {
        'accuracy:': accuracy,
        'precision:': precision,
        'recall:': recall,
        'f1:': f1,
        'confusion_matrix:': cf
    }
    return metrics_dict

