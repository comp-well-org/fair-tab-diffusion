import os
import sys
import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTENC

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

# importing the required files from the parent directory
from lib import load_config, copy_file
from src.evaluate.metrics import evaluate_syn_data, print_metric
from constant import DB_PATH, EXPS_PATH

warnings.filterwarnings('ignore')

def preprocess(data_dir, subset=False):
    xn_train = pd.read_csv(os.path.join(data_dir, 'xn_train.csv'), index_col=0)
    xn_eval = pd.read_csv(os.path.join(data_dir, 'xn_eval.csv'), index_col=0)
    xn_test = pd.read_csv(os.path.join(data_dir, 'xn_test.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    
    if subset:
        xn_train = xn_train.head(1000)
        xn_eval = xn_eval.head(1000)
        xn_test = xn_test.head(1000)
        y_train = y_train.head(1000)
        y_eval = y_eval.head(1000)
        y_test = y_test.head(1000)
    
    # only the first column of y_train, y_eval, y_test is used
    y_train = y_train.iloc[:, 0]
    y_eval = y_eval.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    with open(os.path.join(data_dir, 'desc.json')) as f:
        desc = json.load(f)

    X_train = xn_train.values
    X_val = xn_eval.values
    X_test = xn_test.values
    
    y_train = y_train.values
    y_val = y_eval.values
    y_test = y_test.values
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), desc

class CustomSMOTENC(SMOTENC):
    """Custom SMOTENC."""
    
    def __init__(
        self,
        lam1=0.,
        lam2=1.0,
        *,
        categorical_features,
        sampling_strategy='auto',
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        """Initialize.
        
        Args:
            lam1: lower bound of the step size.
            lam2: upper bound of the step size.
            categorical_features: categorical features.
            sampling_strategy: sampling strategy.
            random_state: random state.
            k_neighbors: number of neighbours.
            n_jobs: number of jobs.
        """
        super().__init__(
            categorical_features=categorical_features,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.lam1 = 0.
        self.lam2 = 1.0
    
    def _make_samples(
        self, x, y_dtype, y_type, nn_data, nn_num, 
        n_samples, step_size=1.0, lam1=0., lam2=1.0,
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(low=self.lam1, high=self.lam2, size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])
    
        x_new = self._generate_samples(x, nn_data, nn_num, rows, cols, steps, y_type=y_type)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return x_new, y_new

def sample_smote(
    x_tvt,
    y_tvt,
    d_num_x,
    d_cat_od_x,
    k_neighbours=5,
    frac_samples=1.0,
    frac_lam_del=0.,
    seed=0,
):
    """Sample SMOTE.
    
    Args:
        x_tvt: `(x_train, x_val, x_test)`.
        y_tvt: `(y_train, y_val, y_test)`.
        d_num_x: number of numerical features.
        d_cat_od_x: number of categorical/ordinal features.
        k_neighbours: number of neighbours.
        frac_samples: fraction of samples to be sampled.
        frac_lam_del: fraction of the step size.
        seed: random seed.
    
    Returns:
        `(x_res, y_res)`.
    """
    lam1 = frac_lam_del / 2
    lam2 = 1.0 - frac_lam_del / 2

    x_train, x_val, x_test = x_tvt
    y_train, y_val, y_test = y_tvt
    
    # cat_features
    strat = {k: int((1 + frac_samples) * np.sum(y_train == k)) for k in np.unique(y_train)}
    cat_features = list(range(d_num_x, d_num_x + d_cat_od_x))

    sm = CustomSMOTENC(
        lam1=lam1,
        lam2=lam2,
        random_state=seed,
        k_neighbors=k_neighbours,
        categorical_features=cat_features,
        sampling_strategy=strat,
    )
    
    x_res, y_res = sm.fit_resample(x_train, y_train)
    y_res = np.expand_dims(y_res, axis=1)
    n_samples = x_train.shape[0]
    x_res = x_res[n_samples:]
    y_res = y_res[n_samples:]
    return x_res, y_res

################################################################################
# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
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
    k_neighbours = model_config['knn']
    
    # message
    print(f'config file: {args.config}')
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
    norm_fn = sio.load(os.path.join(dataset_dir, 'fn.skops'))
    feature_cols = pd.read_csv(os.path.join(dataset_dir, 'x_train.csv'), index_col=0).columns.tolist()
    label_cols = [pd.read_csv(os.path.join(dataset_dir, 'y_train.csv'), index_col=0).columns.tolist()[0]]

    (x_train, x_val, x_test), (y_train, y_val, y_test), desc = preprocess(dataset_dir, subset=False)
    d_num_x = desc['d_num_x']
    d_cat_od_x = desc['d_cat_od_x']

    if args.sample:
        # sampling
        start_time = time.time()
        for i in range(n_seeds):
            random_seed = seed + i
            xn_res, y_res = sample_smote(
                (x_train, x_val, x_test),
                (y_train, y_val, y_test),
                d_num_x,
                d_cat_od_x,
                k_neighbours=k_neighbours,
                frac_samples=1.0,
                frac_lam_del=0.,
                seed=random_seed,
            )
            sample = np.concatenate([xn_res, y_res], axis=1)
            d_numerical = d_num_x
            xn_num = sample[:, :d_numerical]
            x_num = norm_fn.inverse_transform(sample[:, :d_numerical])
            x_cat = sample[:, d_numerical: -1]
            xn_syn = np.concatenate([xn_num, x_cat], axis=1)
            x_syn = np.concatenate([x_num, x_cat], axis=1)
            y_syn = sample[:, -1]
            
            # to dataframe
            xn_syn = pd.DataFrame(xn_syn, columns=feature_cols)
            x_syn = pd.DataFrame(x_syn, columns=feature_cols)
            y_syn = pd.DataFrame(y_syn, columns=label_cols)

            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)
                
            x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            xn_syn.to_csv(os.path.join(synth_dir, 'xn_syn.csv'))
            y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
            print(f'seed: {random_seed}, xn_syn: {xn_syn.shape}, y_syn: {y_syn.shape}')

        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
            time_msg = f'training and sampling time: {end_time - start_time:.2f} seconds with {n_seeds} seeds'
            f.write(time_msg)
        print()
    
    if args.eval:
        # evaluate classifiers trained on synthetic data
        synth_dir_list = []
        for i in range(n_seeds):
            synth_dir = os.path.join(exp_dir, f'synthesis/{seed + i}')
            if os.path.exists(synth_dir):
                synth_dir_list.append(synth_dir)

        sst_col_names = desc['sst_col_names']
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
