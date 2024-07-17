"""Norm."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def ordinal_to_onehot(x: np.array, max_vals: list):
    """Convert ordinal features to one-hot features.
    
    Args:
        x: data of shape `(n_samples, n_features)`.
        max_vals: list of maximum values of ordinal features.
    
    Returns:
        x_onehot: data of shape `(n_samples, sum(max_vals))`.
    """
    x = np.array(x, dtype=np.int32)
    x_onehot = np.zeros((x.shape[0], sum(max_vals)))
    for i in range(len(max_vals)):
        if i == 0:
            x_onehot[:, :max_vals[i]] = np.eye(max_vals[i])[x[:, i]]
        else:
            x_onehot[:, sum(max_vals[:i]):sum(max_vals[:i + 1])] = np.eye(max_vals[i])[x[:, i]]
    return x_onehot


def normalize_x(x_syn: np.array, x_real: np.array, d_num_x: int, max_vals: list = None):
    """Normalize the data.
    
    Args:
        x_syn: synthetic data of shape `(n_samples, n_features)`.
        x_real: real data of shape `(n_samples, n_features)`.
        d_num_x: number of numerical features.
        max_vals: list of maximum values of ordinal features.
    
    Returns:
        x_syn: normalized synthetic data of shape `(n_samples, n_features)`.
        x_real: normalized real data of shape `(n_samples, n_features)`.
    """
    x_num_real = x_real[:, :d_num_x]
    x_cat_real = x_real[:, d_num_x:]
    x_num_syn = x_syn[:, :d_num_x]
    x_cat_syn = x_syn[:, d_num_x:]
    
    if len(x_num_real) > 50000:
        ixs = np.random.choice(len(x_num_real), 50000, replace=False)
        x_num_real = x_num_real[ixs]
        x_cat_real = x_cat_real[ixs]
    
    if len(x_num_syn) > 50000:
        ixs = np.random.choice(len(x_num_syn), 50000, replace=False)
        x_num_syn = x_num_syn[ixs]
        x_cat_syn = x_cat_syn[ixs]
    
    mm = MinMaxScaler().fit(x_num_real)
    x_num_real = mm.transform(x_num_real)
    x_num_syn = mm.transform(x_num_syn)
    
    if x_cat_real is not None:
        x_cat_real = ordinal_to_onehot(x_cat_real, max_vals)
        x_cat_syn = ordinal_to_onehot(x_cat_syn, max_vals)
        x_real = np.concatenate([x_num_real, x_cat_real], axis=1)
        x_syn = np.concatenate([x_num_syn, x_cat_syn], axis=1)
        x_real = np.array(x_real, dtype=np.float32)
        x_syn = np.array(x_syn, dtype=np.float32)
    
    return x_syn, x_real


def _test():
    x_ordinal = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    max_vals = [4, 5, 6]
    x_onehot = ordinal_to_onehot(x_ordinal, max_vals)
    print(x_onehot)

    x = np.random.randint(0, 4, (3, 3))
    x = normalize_x(x, x, 2, [4])
    print(x)


if __name__ == '__main__':
    _test()
