import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH
from lib import load_json, load_config, write_json
from src.evaluate.skmodels import default_sk_clf

warnings.filterwarnings('ignore')

def evaluate_classifier_on_data(dataset, config, save_dir, option='best', dist='original', source='real'):
    assert source in {
        'real',
        'fairsmote',
        'fairtabgan',
        'goggle',
        'great',
        'smote',
        'stasy',
        'tabddpm',
        'tabsyn',
        'fairtabddpm',
    }
    assert option in {'best', 'mean'}
    assert dist in {'uniform', 'original'}
    data_dir = os.path.join(DB_PATH, dataset)
    
    x_train = pd.read_csv(os.path.join(data_dir, 'x_train.csv'), index_col=0)
    c_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_train = c_train.iloc[:, 0]
    
    x_eval = pd.read_csv(os.path.join(data_dir, 'x_eval.csv'), index_col=0)
    c_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_eval = c_eval.iloc[:, 0]
    
    x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'), index_col=0)
    c_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    y_test = c_test.iloc[:, 0]
    
    data_desc = load_json(os.path.join(data_dir, 'desc.json'))
    sst_col_names = data_desc['sst_col_names']
    
    # replace sensitive attributes with random categorical values
    x_train_rand = x_train.copy()
    for col in sst_col_names:
        n_unq = x_train[col].nunique()
        x_train_rand[col] = np.random.choice(n_unq, x_train.shape[0])
        # print(x_train_rand[col].value_counts())
    
    # evaluate classifiers trained on real data
    metric = {}
    sk_clf_lst = config['classifiers'][option]
    seed = config['exp']['seed']
    n_seeds = config['exp']['n_seeds']
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
            for s_col in sst_col_names:
                dprs[flag][s_col] = []
                eors[flag][s_col] = []
        
        for i in range(n_seeds):
            if source == 'real':
                random_seed = seed + i
                print(f'working on {clf_choice} with seed {random_seed} on real data')
                clf = default_sk_clf(clf_choice, seed=random_seed)
                if dist == 'uniform':
                    clf.fit(x_train_rand, y_train)
                elif dist == 'original':
                    clf.fit(x_train, y_train)
            else: 
                session = config['methods'][source]['session']
                random_seed = seed + i
                print(f'working on {clf_choice} with seed {random_seed} on synthetic data')
                synth_dir = os.path.join(EXPS_PATH, dataset, source, session, 'synthesis', f'{random_seed}')
                x_syn = pd.read_csv(os.path.join(synth_dir, 'x_syn.csv'), index_col=0)
                c_syn = pd.read_csv(os.path.join(synth_dir, 'y_syn.csv'), index_col=0)
                y_syn = c_syn.iloc[:, 0]
                # replace sensitive attributes with random categorical values
                x_syn_rand = x_syn.copy()
                for col in sst_col_names:
                    n_unq = x_syn[col].nunique()
                    x_syn_rand[col] = np.random.choice(n_unq, x_syn.shape[0])
                    # print(x_syn_rand[col].value_counts())
                
                clf = default_sk_clf(clf_choice)
                if dist == 'uniform':
                    clf.fit(x_syn_rand, y_syn)
                elif dist == 'original':
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
                for s_col in sst_col_names:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--config', type=str, default='./assess.toml')
    parser.add_argument(
        '--source', type=str, default='real', 
        choices=[
            'real', 'fairsmote', 'fairtabgan', 'goggle', 'great', 
            'smote', 'stasy', 'tabddpm', 'tabsyn', 'fairtabddpm',
        ],
    )
    parser.add_argument('--option', type=str, default='best', choices=['best', 'mean'])
    parser.add_argument('--dist', type=str, default='original', choices=['uniform', 'original'])

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # divider
    print(f'evaluating classifiers on {args.dataset} dataset with {args.source} data')
    print(f'using {args.option} classifier and {args.dist} distribution on sensitive attributes')
    print('-' * 80)
    
    # save results
    save_dir = f'eval/learning/{args.dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    option = args.option
    dist = args.dist
    source = args.source

    metrics = evaluate_classifier_on_data(
        args.dataset, config, save_dir, option=option, 
        dist=dist, source=source,
    )
    file_path = os.path.join(save_dir, f'{option}_{dist}_{source}.json')
    write_json(metrics, file_path)
    print(f'saved results to {file_path}')
    print('done!')
    print()
    
if __name__ == '__main__':
    main()
