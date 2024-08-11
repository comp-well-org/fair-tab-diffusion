import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from .skmodels import default_sk_clf
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

def evaluate_syn_data(data_dir: str, exp_dir: str, synth_dir_list: list, sk_clf_lst: list, sens_cols: list):
    x_train = pd.read_csv(os.path.join(data_dir, 'x_train.csv'), index_col=0)
    c_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_train = c_train.iloc[:, 0]
    
    x_eval = pd.read_csv(os.path.join(data_dir, 'x_eval.csv'), index_col=0)
    c_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_eval = c_eval.iloc[:, 0]
    
    x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'), index_col=0)
    c_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    y_test = c_test.iloc[:, 0]
    
    # evaluate classifiers trained on synthetic data
    metric = {}
    for clf_choice in sk_clf_lst:
        aucs = {
            'Train': [],
            'Validation': [],
            'Test': [],
        }
        dprs = {
            'Train': {},
            'Validation': {},
            'Test': {},
        }
        eors = {
            'Train': {},
            'Validation': {},
            'Test': {},
        }
        for flag in ['Train', 'Validation', 'Test']:
            for s_col in sens_cols:
                dprs[flag][s_col] = []
                eors[flag][s_col] = []
        
        for synth_dir in synth_dir_list:
            # read synthetic data
            x_syn = pd.read_csv(os.path.join(synth_dir, 'x_syn.csv'), index_col=0)
            c_syn = pd.read_csv(os.path.join(synth_dir, 'y_syn.csv'), index_col=0)
            y_syn = c_syn.iloc[:, 0]
            
            # train classifier
            clf = default_sk_clf(clf_choice)
            clf.fit(x_syn, y_syn)
            
            # training set
            for flag, xx, yy, cc in zip(
                ['Train', 'Validation', 'Test'], 
                [x_train, x_eval, x_test], 
                [y_train, y_eval, y_test], 
                [c_train, c_eval, c_test],
            ):
                y_pred_label = clf.predict(xx)
                y_pred_proba = clf.predict_proba(xx)[:, 1]
                aucs[flag].append(roc_auc_score(yy, y_pred_proba))
                for s_col in sens_cols:
                    ss = cc[s_col]
                    dpr = demographic_parity_ratio(yy, y_pred_label, sensitive_features=ss)
                    eor = equalized_odds_ratio(yy, y_pred_label, sensitive_features=ss)
                    dprs[flag][s_col].append(dpr)
                    eors[flag][s_col].append(eor)
        
        metric[clf_choice] = {
            'AUC': aucs,
            'DPR': dprs,
            'EOR': eors,
        }
    return metric
