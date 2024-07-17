"""Standard tabular data formats."""

import os
import sklearn
import json
import shutil
import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from typing import Dict

class DataDesc:
    """Tabular data description."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        n_channels: int,
        x_norm_type: str,
        col_names: list[str],
        sst_col_names: list[str],
        sst_col_indices: list[int],
        d_num_x: int,
        d_cat_od_x: int,
        d_cat_oh_x: int,
        d_od_x: int,
        d_oh_x: int,
        n_unq_cat_od_x_lst: list[int],
        cat_od_x_fn: dict[str, list[str]],
        n_unq_y: int,
        cat_od_y_fn: dict[int, str],
        n_unq_c_lst: list[int],
    ) -> None:
        """Initialize.
        
        Args:
            dataset_name: name of the dataset.
            dataset_type: type of the dataset.
            n_channels: number of channels.
            x_norm_type: type of x normalization.
            col_names: names of all columns.
            sst_col_names: names of sensitive columns.
            sst_col_indices: indices of sensitive columns.
            d_num_x: number of numerical features.
            d_cat_od_x: number of ordinal categorical features.
            d_cat_oh_x: number of one-hot categorical features.
            d_od_x: number of ordinal and numerical features.
            d_oh_x: number of one-hot and numerical features.
            n_unq_cat_od_x_lst: number of unique values for each ordinal categorical feature.
            cat_od_x_fn: mapping function for ordinal categorical features.
            n_unq_y: number of unique values for the outcome.
            cat_od_y_fn: mapping function for the outcome.
            n_unq_c_lst: number of unique values for each sensitive feature.
        """
        self._desc = {}
        self._desc['dataset_name'] = dataset_name
        self._desc['dataset_type'] = dataset_type
        self._desc['n_channels'] = n_channels
        self._desc['x_norm_type'] = x_norm_type
        self._desc['col_names'] = col_names
        self._desc['sst_col_names'] = sst_col_names
        self._desc['sst_col_indices'] = sst_col_indices
        self._desc['d_num_x'] = d_num_x
        self._desc['d_cat_od_x'] = d_cat_od_x
        self._desc['d_cat_oh_x'] = d_cat_oh_x
        self._desc['d_od_x'] = d_od_x
        self._desc['d_oh_x'] = d_oh_x
        self._desc['n_unq_cat_od_x_lst'] = n_unq_cat_od_x_lst
        self._desc['cat_od_x_fn'] = cat_od_x_fn
        self._desc['n_unq_y'] = n_unq_y
        self._desc['cat_od_y_fn'] = cat_od_y_fn
        self._desc['n_unq_c_lst'] = n_unq_c_lst
    
    @property
    def desc(self) -> str:
        return self._desc

def norm_x(
    x_train_val_test: tuple[np.ndarray, np.ndarray, np.ndarray], 
    x_norm_type: str, 
    seed: int = 2023,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, callable]:
    train, val, test = x_train_val_test
    if x_norm_type == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif x_norm_type == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif x_norm_type == 'quantile':
        slices = 30
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(train.shape[0] // slices, 1000), 10),
            subsample=10 ** 9,
            random_state=seed,
        )
    elif x_norm_type == 'none':
        normalizer = sklearn.preprocessing.FunctionTransformer()
    
    normalizer.fit(train)
    normed_train = normalizer.transform(train)
    normed_val = normalizer.transform(val)
    normed_test = normalizer.transform(test)
    return normed_train, normed_val, normed_test, normalizer

def train_val_test_split(
    x_all: np.array, y_all: np.array, split_size: tuple[int, int], seed: int = 2023,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x_train, x_rest, y_train, y_rest = train_test_split(
        x_all, y_all, test_size=split_size[0], random_state=seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=split_size[1], random_state=seed,
    )
    return (x_train, x_val, x_test), (y_train, y_val, y_test)

def save_train_val_test_norm(
    db_dir: str,
    desc: Dict,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    xn_train: np.ndarray,
    xn_val: np.ndarray,
    xn_test: np.ndarray,
    normalizer: callable, 
) -> None:
    np.save(os.path.join(db_dir, 'x-all.npy'), x_all)
    np.save(os.path.join(db_dir, 'y-all.npy'), y_all)
    np.save(os.path.join(db_dir, 'x-train.npy'), x_train)
    np.save(os.path.join(db_dir, 'y-train.npy'), y_train)
    np.save(os.path.join(db_dir, 'x-val.npy'), x_val)
    np.save(os.path.join(db_dir, 'y-val.npy'), y_val)
    np.save(os.path.join(db_dir, 'x-test.npy'), x_test)
    np.save(os.path.join(db_dir, 'y-test.npy'), y_test)
    np.save(os.path.join(db_dir, 'xn-train.npy'), xn_train)
    np.save(os.path.join(db_dir, 'xn-val.npy'), xn_val)
    np.save(os.path.join(db_dir, 'xn-test.npy'), xn_test)
    sio.dump(normalizer, os.path.join(db_dir, 'fn.skops'))
    with open(os.path.join(db_dir, 'desc.json'), 'w') as f:
        json.dump(desc, f, indent=4)

def process_pipeline(
    dataset_name: str,
    x_norm_type: str,
    split_size: tuple[float, float],
    seed: int,
    save_dir: str,
    override: bool,
    features: pd.DataFrame,
    labels: pd.Series,
    cat_columns: list[str],
    sst_columns: list[str],
    num_columns: list[str],
):
    num_features = features[num_columns]
    cat_features = features[cat_columns]
    
    # convert categorical features to numerical
    ordinal_encoder = OrdinalEncoder()
    ordinal_cat_features = ordinal_encoder.fit_transform(cat_features)
    
    # concatenate numerical and categorical features
    x_all = np.concatenate((num_features, ordinal_cat_features), axis=1)
    
    # reshape labels to (n, 1)
    y_all = np.array(labels).reshape(-1, 1)

    # summarize information about data
    col_names = num_columns + cat_columns
    sst_col_indices = [col_names.index(name) for name in sst_columns]
    d_num_x = len(num_columns)
    
    # combine outcomes and sensitive features
    y_all = np.concatenate((y_all, x_all[:, sst_col_indices]), axis=1)
    
    # mapping relation
    cat_ordinal_mapping = {}
    for col_name, cat in zip(cat_columns, ordinal_encoder.categories_):
        cat_ordinal_mapping[col_name] = list(cat)
    n_unq_y = len(np.unique(labels))
    n_unq_sst_lst = [len(cat_ordinal_mapping[name]) for name in sst_columns]
    n_unq_c_lst = [n_unq_y] + n_unq_sst_lst

    # number of classes for each categorical feature
    n_classes_lst = [len(cate) for cate in ordinal_encoder.categories_]

    # description of the dataset
    data_desc = DataDesc(
        dataset_name=dataset_name,
        dataset_type='tabular',
        n_channels=1,
        x_norm_type=x_norm_type,
        col_names=col_names,
        sst_col_names=sst_columns,
        sst_col_indices=sst_col_indices,
        d_num_x=d_num_x,
        d_cat_od_x=len(cat_columns),
        d_cat_oh_x=sum(n_classes_lst),
        d_od_x=len(col_names),
        d_oh_x=len(num_columns) + sum(n_classes_lst),
        n_unq_cat_od_x_lst=n_classes_lst,
        cat_od_x_fn=cat_ordinal_mapping,
        n_unq_y=n_unq_y,
        cat_od_y_fn=None,
        n_unq_c_lst=n_unq_c_lst,
    )
    
    # split dataset
    (x_train, x_val, x_test), (y_train, y_val, y_test) = train_val_test_split(
        x_all, y_all, split_size=split_size, seed=seed,
    )
    
    # normalize x numerical features
    x_num_train_norm, x_num_val_norm, x_num_test_norm, normalizer = norm_x(
        (x_train[:, :d_num_x], x_val[:, :d_num_x], x_test[:, :d_num_x]),
        x_norm_type=x_norm_type,
        seed=seed,
    )
    xn_train = np.concatenate((x_num_train_norm, x_train[:, d_num_x:]), axis=1)
    xn_val = np.concatenate((x_num_val_norm, x_val[:, d_num_x:]), axis=1)
    xn_test = np.concatenate((x_num_test_norm, x_test[:, d_num_x:]), axis=1)
    
    # save dataset
    if save_dir is not None:
        db_dir = os.path.join(save_dir, dataset_name)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        else:
            if override:
                shutil.rmtree(db_dir)
                os.makedirs(db_dir)
            else:
                raise FileExistsError(f'{db_dir} already exists.')
        # save files
        save_train_val_test_norm(
            db_dir=db_dir,
            desc=data_desc.desc,
            x_all=x_all,
            y_all=y_all,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            xn_train=xn_train,
            xn_val=xn_val,
            xn_test=xn_test,
            normalizer=normalizer,
        )
