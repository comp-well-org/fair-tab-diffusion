"""SMOTE."""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTENC

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
    return x_res, y_res

################################################################################
# main
def main():    
    # TODO: configs
    dataname = 'adult'
    
    # data
    data_dir = f'/rdf/db/public-tabular-datasets/{dataname}/'
    (x_train, x_val, x_test), (y_train, y_val, y_test), desc = preprocess(data_dir, subset=True)
    d_num_x = desc['d_num_x']
    d_cat_od_x = desc['d_cat_od_x']
    
    x_res, y_res = sample_smote(
        (x_train, x_val, x_test),
        (y_train, y_val, y_test),
        d_num_x,
        d_cat_od_x,
        k_neighbours=5,
        frac_samples=1.0,
        frac_lam_del=0.,
        seed=0,
    )
    print(x_res.shape, y_res.shape)

if __name__ == '__main__':
    main()
