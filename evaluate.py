import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.evaluate.skmodels import default_sk_clf
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

def evaluate_syn_data(data_dir, exp_dir, seed, n_seeds, sk_clf_lst, sens='sex'):
    x_eval = pd.read_csv(os.path.join(data_dir, 'x_eval.csv'), index_col=0)
    c_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_eval = c_eval.iloc[:, 0]
    sensitive = c_eval[sens]
    
    # evaluate classifiers trained on synthetic data
    metric = {}
    for clf_choice in sk_clf_lst:
        aucs = []
        dprs = []
        eors = []
        for i in range(n_seeds):
            random_seed = seed + i
            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            
            # read synthetic data
            x_syn = pd.read_csv(os.path.join(synth_dir, 'x_syn.csv'), index_col=0)
            c_syn = pd.read_csv(os.path.join(synth_dir, 'y_syn.csv'), index_col=0)
            y_syn = c_syn.iloc[:, 0]
            
            # train classifier
            clf = default_sk_clf(clf_choice, random_seed)
            clf.fit(x_syn, y_syn)
            y_pred = clf.predict_proba(x_eval)[:, 1]
            aucs.append(roc_auc_score(y_eval, y_pred))
            
            # fairness metrics
            y_pred_label = clf.predict(x_eval)
            dpr = demographic_parity_ratio(y_eval, y_pred_label, sensitive_features=sensitive)
            eor = equalized_odds_ratio(y_eval, y_pred_label, sensitive_features=sensitive)
            dprs.append(dpr)
            eors.append(eor)
        
        metric[clf_choice] = {
            'AUC': (np.mean(aucs), np.std(aucs)),
            'DPR': (np.mean(dprs), np.std(dprs)),
            'EOR': (np.mean(eors), np.std(eors)),
        }
    return metric
