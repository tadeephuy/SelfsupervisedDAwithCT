import pandas as pd
import numpy as np
from sklearn import metrics


def split_src_tgt(x):
    bs = x.shape[0]
    return x[:bs//2], x[bs//2:]

def f1_score(preds, targets, sigmoid=True, thresh=0.5, average='micro', idx=None, thresh_target=None):
    if sigmoid: preds = 1/(1 + np.exp(-preds))
    preds = (preds >= thresh).astype(np.uint8)
    if thresh_target is not None:
        targets = (targets >= thresh_target).astype(np.uint8)
    if idx is not None:
        return metrics.fbeta_score(y_true=targets, y_pred=preds, beta=1, average=None)[idx]
    return metrics.fbeta_score(y_true=targets, y_pred=preds, beta=1, average=average)

def accuracy(preds, targets):
    return metrics.accuracy_score(targets, preds)

def calculate_metric(preds_dict, labels_dict):
    src_class_f1 = f1_score(preds_dict['p_src'], labels_dict['y_src'], idx=0, thresh_target=0.5)
    src_aux_accuracy = accuracy(preds_dict['aux_pred_src'], labels_dict['a_src'])
    tgt_aux_accuracy = accuracy(preds_dict['aux_pred_tgt'], labels_dict['a_tgt'])

    return {
        'src_class_f1': src_class_f1,
        'src_aux_accuracy': src_aux_accuracy,
        'tgt_aux_accuracy': tgt_aux_accuracy
    }

def aggregate_logs(logs):
    df = pd.DataFrame(logs)

    losses = pd.DataFrame(list(df['losses'].values))
    losses_results = losses.mean().to_dict()
    
    preds = pd.DataFrame(list(df['preds'].values))
    pred_results = {k: None for k in preds.columns}
    for pred in preds.columns:
        pred_results[pred] = np.concatenate(preds[pred].apply(np.array).values, axis=0)

    result = {
        'losses': losses_results,
        'preds': pred_results,
    }

    if 'labels' in df.columns:
        labels = pd.DataFrame(list(df['labels'].values))
        labels_results = {k: None for k in labels.columns}
        for label in labels.columns:
            labels_results[label] = np.concatenate(labels[label].apply(np.array).values, axis=0)

        result['labels'] = labels_results
        result['metrics'] = calculate_metric(result['preds'], result['labels'])

    return result