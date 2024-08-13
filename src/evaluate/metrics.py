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
            # if y_syn has only one unique value, skip training
            if len(y_syn.unique()) == 1:
                for flag in ['Train', 'Validation', 'Test']:
                    aucs[flag].append(0.0)
                    for s_col in sens_cols:
                        dprs[flag][s_col].append(0.0)
                        eors[flag][s_col].append(0.0)
            else:
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

def print_metric(metric):
    val_auc_list = metric['CatBoost']['AUC']['Validation']     
    test_auc_list = metric['CatBoost']['AUC']['Test']
    
    # keep only float values
    val_auc_list = [x for x in val_auc_list if isinstance(x, float)]
    test_auc_list = [x for x in test_auc_list if isinstance(x, float)]
    
    val_auc = sum(val_auc_list) / len(val_auc_list)   
    test_auc = sum(test_auc_list) / len(test_auc_list)
    print(f'Validation AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
    
    val_dpr_dict = metric['CatBoost']['DPR']['Validation']
    test_dpr_dict = metric['CatBoost']['DPR']['Test']
    
    val_eor_dict = metric['CatBoost']['EOR']['Validation']
    test_eor_dict = metric['CatBoost']['EOR']['Test']
    for key in val_dpr_dict.keys():
        val_dpr_list = val_dpr_dict[key]
        test_dpr_list = test_dpr_dict[key]
        
        # keep only float values
        val_dpr_list = [x for x in val_dpr_list if isinstance(x, float)]
        test_dpr_list = [x for x in test_dpr_list if isinstance(x, float)]
        
        val_dpr = sum(val_dpr_list) / len(val_dpr_list)
        test_dpr = sum(test_dpr_list) / len(test_dpr_list)
        print(f'Validation DPR ({key}): {val_dpr:.4f}, Test DPR ({key}): {test_dpr:.4f}')
    
    for key in val_eor_dict.keys():
        val_eor_list = val_eor_dict[key]
        test_eor_list = test_eor_dict[key]
        
        # keep only float values
        val_eor_list = [x for x in val_eor_list if isinstance(x, float)]
        test_eor_list = [x for x in test_eor_list if isinstance(x, float)]
        
        val_eor = sum(val_eor_list) / len(val_eor_list)
        test_eor = sum(test_eor_list) / len(test_eor_list)
        print(f'Validation EOR ({key}): {val_eor:.4f}, Test EOR ({key}): {test_eor:.4f}')
