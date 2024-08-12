import sklearn
import pandas as pd

class TabDataDesc:    
    def __init__(
        self,
        dataset_name: str,
        dataset_type: str,
        n_channels: int,
        x_norm_type: str,
        col_names: list[str],
        label_col_name: str,
        num_col_names: list[str],
        cat_col_names: list[str],
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
        train_num: int,
        eval_num: int,
        test_num: int,
    ) -> None:
        """Initialize an object for dataset description.
        
        Args:
            dataset_name: name of the dataset.
            dataset_type: type of the dataset.
            n_channels: number of channels.
            x_norm_type: type of x normalization.
            col_names: names of all columns.
            label_col_name: name of the label column.
            num_col_names: names of numerical columns.
            cat_col_names: names of categorical columns.
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
            train_num: number of training samples.
            eval_num: number of validation samples.
            test_num: number of test samples.
        """
        self._desc = {}
        self._desc['dataset_name'] = dataset_name
        self._desc['dataset_type'] = dataset_type
        self._desc['n_channels'] = n_channels
        self._desc['x_norm_type'] = x_norm_type
        self._desc['col_names'] = col_names
        self._desc['label_col_name'] = label_col_name
        self._desc['num_col_names'] = num_col_names
        self._desc['cat_col_names'] = cat_col_names
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
        self._desc['train_num'] = train_num
        self._desc['eval_num'] = eval_num
        self._desc['test_num'] = test_num
    
    @property
    def desc(self) -> str:
        return self._desc

def norm_tab_x_df(
    x_train_eval_test: tuple, 
    x_norm_type: str, 
    seed: int = 2023,
) -> tuple:
    train, eval, test = x_train_eval_test
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
    normed_eval = normalizer.transform(eval)
    normed_test = normalizer.transform(test)
    normed_train = pd.DataFrame(normed_train, columns=train.columns, index=train.index)
    normed_eval = pd.DataFrame(normed_eval, columns=eval.columns, index=eval.index)
    normed_test = pd.DataFrame(normed_test, columns=test.columns, index=test.index)
    return normed_train, normed_eval, normed_test, normalizer
