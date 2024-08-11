import os
import sys
import json
import time
import warnings
import argparse
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples
from imblearn.over_sampling import SMOTENC

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from lib import load_config, copy_file, load_json
from src.evaluate.metrics import evaluate_syn_data, print_metric

warnings.filterwarnings('ignore')

class FairBalance:
    def __init__(
        self, data, features, continous_features, 
        drop_features, sensitive_attribute, target, 
        cluster_algo='kmeans', ratio=0.25, knn=5,
    ):
        self.data = data.copy()
        self.continous_features = continous_features
        self.sensitive_attribute = sensitive_attribute
        self.features = features
        self.drop_features = drop_features
        self.target = target
        self.cluster_algo = cluster_algo
        self.ratio = ratio
        self.knn = knn

    def fit(self):
        self.cluster()
        self.filter()

    def cluster(self):
        scaler = MinMaxScaler()
        X = self.data.drop(self.sensitive_attribute + [self.target], axis=1)
        X[self.continous_features] = scaler.fit_transform(X[self.continous_features])

        if self.cluster_algo == 'kmeans':
            model = KMeans()
        elif self.cluster_algo == 'agg':
            model = AgglomerativeClustering()
        elif self.cluster_algo == 'spe':
            model = SpectralClustering()
        else:
            model = self.cluster_algo

        max_s = -np.inf
        for k in range(2, 10):
            model = model.set_params(n_clusters=k)
            model.fit(X)
            groups = model.labels_
            s_score = silhouette_score(X, groups)
            score_list = silhouette_samples(X, groups)
            if s_score > max_s:
                max_s = s_score
                best_clist = groups
                best_slist = score_list
        # print(f'cluster the original dataset into {best_k} clusters...')
        self.data['score'] = best_slist
        self.data['group'] = best_clist
        
    def filter(self):
        scores = self.data['score'].tolist()
        s_rank = np.sort(scores)
        idx = int(len(s_rank)*self.ratio)
        threshold = s_rank[idx]
        # print(f'removing {idx} samples from the original dataset...')
        self.X_clean = self.data[self.data['score'] > threshold]

    def new_smote(self, dfi, seed=42):
        label = self.target
        categorical_features = list(set(dfi.keys().tolist()) - set(self.continous_features))
        categorical_loc = [dfi.columns.get_loc(c) for c in categorical_features if c in dfi.keys()]

        # count the class distribution 
        min_y = dfi[label].value_counts().idxmin()
        max_y = dfi[label].value_counts().idxmax()
        min_X = dfi[dfi[label] == min_y]
        max_X = dfi[dfi[label] == max_y]
        ratio = len(max_X) - len(min_X)
        
        nbrs = NearestNeighbors(
            n_neighbors=min(self.knn, len(dfi)), algorithm='auto',
        ).fit(dfi[self.features])
        dfs = []
        for j in range(len(min_X)):
            dfj = min_X.iloc[j]
            nn_list = nbrs.kneighbors(
                np.array(dfj[self.features]).reshape(1, -1), 
                return_distance=False,
            )[0]
            df_nn = dfi.iloc[nn_list]
            dfs.append(df_nn)
        df_nns = pd.concat(list(dfs), ignore_index=True).drop_duplicates()   

        X_temp = pd.concat([df_nns, min_X], ignore_index=True)
        y_temp = list(np.repeat(1, len(df_nns))) + list(np.repeat(0, len(min_X)))
        
        min_k = max(1, min(self.knn, len(df_nns)-1))
        sm = SMOTENC(
            categorical_features=categorical_loc, random_state=seed, 
            sampling_strategy={1: len(df_nns)+ratio, 0: len(min_X)}, 
            k_neighbors=min_k,
        )
        Xi_res, yi_res = sm.fit_resample(X_temp, y_temp)
        df_res = pd.DataFrame(Xi_res, columns=dfi.keys().tolist())
        df_add = df_res.iloc[len(X_temp):]
        df_add[label] = min_y
        df_new = pd.concat([dfi, df_add], ignore_index=True)
        return df_new

    def generater(self, seed=42):
        dfs = []
        groups = list(self.X_clean['group'].unique())
        for i in groups:
            dfi = self.X_clean[self.X_clean['group'] == i].drop(['group', 'score'], axis=1)
            if (len(dfi[self.target].unique()) == 1 or len(dfi) == 0):
                continue
            Xi_res = self.new_smote(dfi, seed)
            dfs.append(Xi_res)
        X_cres = pd.concat(list(dfs), ignore_index=True)
        return X_cres[self.features], X_cres[self.target]

def balancing(x_train, target, knn, sensitive_attribute, features, drop_features, continous_features, seed=42):
    fcb = FairBalance(
        x_train, features, 
        continous_features, drop_features, 
        sensitive_attribute, target, knn=knn,
    )
    fcb.fit()
    X_balanced, y_balanced = fcb.generater(seed=seed)
    return X_balanced, y_balanced

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    parser.add_argument('--train', action='store_true', help='training', default=True)
    parser.add_argument('--sample', action='store_true', help='sampling', default=True)
    parser.add_argument('--eval', action='store_true', help='evaluation', default=True)
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # configs
    exp_config = config['exp']
    data_config = config['data']
    model_config = config['model']
    sample_config = config['sample']
    eval_config = config['eval']
    
    seed = exp_config['seed']
    n_seeds = sample_config['n_seeds']
    knn = model_config['knn']
    
    # message
    print(json.dumps(config, indent=4))
    print('-' * 80)
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        args.exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        args.config,
    )
    
    # data
    dataset_dir = os.path.join(data_config['path'], data_config['name'])
    data_desc = load_json(os.path.join(dataset_dir, 'desc.json'))
    x_train = pd.read_csv(f'{dataset_dir}/x_train.csv', index_col=0)
    y_train = pd.read_csv(f'{dataset_dir}/y_train.csv', index_col=0)
    label_col = y_train.columns.tolist()[0]
    y_train = y_train[[label_col]]
    
    data_train = pd.concat([x_train, y_train], axis=1)
    feature_cols = pd.read_csv(os.path.join(dataset_dir, 'x_train.csv'), index_col=0).columns.tolist()
    d_num_x = data_desc['d_num_x']
    x_num_cols = feature_cols[:d_num_x]
    sst_col_names = data_desc['sst_col_names']

    drop_features = []
    features = list(set(data_train.keys().tolist()) - set(drop_features + [label_col]))

    # model
    fcb = FairBalance(
        data_train, features, 
        x_num_cols, drop_features,
        sst_col_names, label_col, knn=knn,
    )
    
    if args.train:
        # train
        start_time = time.time()
        fcb.fit()
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
            time_msg = f'training time: {end_time - start_time:.2f} seconds'
            f.write(time_msg)

    if args.sample:
        # sampling
        start_time = time.time()
        for i in range(n_seeds):
            random_seed = seed + i

            x_syn_balanced, y_syn_balanced = fcb.generater(seed=random_seed)
            # reordering the columns of the balanced dataset
            x_syn_balanced = x_syn_balanced[feature_cols]
            
            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)
            
            x_syn_balanced.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            y_syn_balanced.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
            print(f'seed: {random_seed}, x_syn: {x_syn_balanced.shape}, y_syn: {y_syn_balanced.shape}')
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'a') as f:
            time_msg = f'\nsampling time: {end_time - start_time:.2f} seconds with {n_seeds} seeds'
            f.write(time_msg)
            
    if args.eval:
        # evaluate classifiers trained on synthetic data
        synth_dir_list = []
        for i in range(n_seeds):
            synth_dir = os.path.join(exp_dir, f'synthesis/{seed + i}')
            if os.path.exists(synth_dir):
                synth_dir_list.append(synth_dir)

        sst_col_names = data_desc['sst_col_names']
        metric = evaluate_syn_data(
            data_dir=os.path.join(data_config['path'], data_config['name']),
            exp_dir=exp_dir,
            synth_dir_list=synth_dir_list,
            sk_clf_lst=eval_config['sk_clf_choice'],
            sens_cols=sst_col_names,
        )

        with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
            json.dump(metric, f, indent=4)

        # print metric
        print_metric(metric)

if __name__ == '__main__':
    main()
