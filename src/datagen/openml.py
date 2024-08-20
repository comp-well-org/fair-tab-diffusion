import os
import json
import skops.io as sio
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict
from .tabular import TabDataDesc, norm_tab_x_df

UCI_ADULT_ID = 1590
COMPASS_ID = 44053
UCI_GERMAN_CREDIT_ID = 46116
BANK_MARKETING_ID = 44234
LAW_SCHOOL_ID = 43890

def process_dataset(
    dataset_name: str,
    dataset_type: str,
    n_channels: int,
    data_all: pd.DataFrame,
    data_all_cp: pd.DataFrame,
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    x_norm_type: str,
    col_names: list,
    label_col_name: str,
    sst_columns: list,
    d_num_x: int,
    cat_columns: list,
    num_columns: list,
    n_classes_lst: list,
    sst_col_indices: list,
    cat_ordinal_mapping: dict,
    cat_od_y_fn: dict,
    cat_ord_enc: OrdinalEncoder,
    label_ord_enc: OrdinalEncoder,
    n_unq_y: int,
    n_unq_c_lst: list,
    ratios: tuple,
    seed: int = 42,
    dir_path: str = None,
):
    ratio1, ratio2 = ratios
    x_train, x_rest, y_train, y_rest = train_test_split(
        feature_df, label_df, test_size=ratio1, random_state=seed,
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_rest, y_rest, test_size=ratio2, random_state=seed,
    )
    x_num_train_norm, x_num_eval_norm, x_num_test_norm, normalizer = norm_tab_x_df(
        (x_train.iloc[:, :d_num_x], x_eval.iloc[:, :d_num_x], x_test.iloc[:, :d_num_x]),
        x_norm_type=x_norm_type,
        seed=seed,
    )
    xn_train = pd.concat([x_num_train_norm, x_train.iloc[:, d_num_x:]], axis=1)
    xn_eval = pd.concat([x_num_eval_norm, x_eval.iloc[:, d_num_x:]], axis=1)
    xn_test = pd.concat([x_num_test_norm, x_test.iloc[:, d_num_x:]], axis=1)
    
    data_train = data_all_cp.loc[x_train.index]
    data_eval = data_all_cp.loc[x_eval.index]
    data_test = data_all_cp.loc[x_test.index]
    
    d_types = {col: str(x_train[col].dtype) for col in x_train.columns}
    d_types[label_col_name] = str(y_train[label_col_name].dtype)
    
    unreordered_cols = list(data_all_cp.columns)
    num_col_idx = [unreordered_cols.index(name) for name in num_columns]
    cat_col_idx = [unreordered_cols.index(name) for name in cat_columns]
    target_col_idx = [unreordered_cols.index(label_col_name)]

    data_desc = TabDataDesc(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        n_channels=n_channels,
        x_norm_type=x_norm_type,
        col_names=col_names,
        label_col_name=label_col_name,
        num_col_names=num_columns,
        cat_col_names=cat_columns,
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
        cat_od_y_fn=cat_od_y_fn,
        n_unq_c_lst=n_unq_c_lst,
        train_num=len(x_train),
        eval_num=len(x_eval),
        test_num=len(x_test),
        d_types=d_types,
        name=dataset_name,
        task_type='binclass',
        header='infer',
        column_names=col_names + [label_col_name],
        num_col_idx=num_col_idx,
        cat_col_idx=cat_col_idx,
        target_col_idx=target_col_idx,
        file_type='csv',
        train_path=f'{dir_path}/d_train.csv',
        eval_path=f'{dir_path}/d_eval.csv',
        test_path=f'{dir_path}/d_test.csv',
    )
    
    if dir_path:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        data_all_cp.to_csv(f'{dir_path}/d_all.csv', index=True)
        data_train.to_csv(f'{dir_path}/d_train.csv', index=True)
        data_eval.to_csv(f'{dir_path}/d_eval.csv', index=True)
        data_test.to_csv(f'{dir_path}/d_test.csv', index=True)
        feature_df.to_csv(f'{dir_path}/x_all.csv', index=True)
        label_df.to_csv(f'{dir_path}/y_all.csv', index=True)
        x_train.to_csv(f'{dir_path}/x_train.csv', index=True)
        x_eval.to_csv(f'{dir_path}/x_eval.csv', index=True)
        x_test.to_csv(f'{dir_path}/x_test.csv', index=True)
        y_train.to_csv(f'{dir_path}/y_train.csv', index=True)
        y_eval.to_csv(f'{dir_path}/y_eval.csv', index=True)
        y_test.to_csv(f'{dir_path}/y_test.csv', index=True)
        xn_train.to_csv(f'{dir_path}/xn_train.csv', index=True)
        xn_eval.to_csv(f'{dir_path}/xn_eval.csv', index=True)
        xn_test.to_csv(f'{dir_path}/xn_test.csv', index=True)
        sio.dump(normalizer, f'{dir_path}/fn.skops')
        sio.dump(cat_ord_enc, f'{dir_path}/cat_encoder.skops')
        sio.dump(label_ord_enc, f'{dir_path}/label_encoder.skops')
        with open(f'{dir_path}/desc.json', 'w') as f:
            json.dump(data_desc.desc, f, indent=4)
        
        with open(f'{dir_path}/size.json', 'w') as f:
            json.dump({'n_train': len(x_train), 'n_eval': len(x_eval), 'n_test': len(x_test)}, f, indent=4)
    
    return {
        'desc': data_desc.desc,
        'data_all': data_all,
        'x_all': feature_df,
        'y_all': label_df,
        'x_train': x_train,
        'x_eval': x_eval,
        'x_test': x_test,
        'y_train': y_train,
        'y_eval': y_eval,
        'y_test': y_test,
        'xn_train': xn_train,
        'xn_eval': xn_eval,
        'xn_test': xn_test,
        'normalizer': normalizer,
        'cat_encoder': cat_ord_enc,
        'label_encoder': label_ord_enc,
    }

def get_vars_from_data(
    features: pd.DataFrame,
    labels: pd.Series,
    sst_columns: list,
    cat_columns: list,
    num_columns: list,
    label_col: str,
    label_ord_enc: OrdinalEncoder,
):
    num_features = features[num_columns]
    cat_features = features[cat_columns].astype(str)
    cat_ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    cat_ord_features = cat_ord_enc.fit_transform(cat_features)
    cat_ord_features = pd.DataFrame(cat_ord_features, columns=cat_columns, index=cat_features.index)

    feature_df = pd.concat([num_features, cat_ord_features], axis=1)
    label_df = pd.DataFrame(labels, columns=[label_col])
    label_df[sst_columns] = feature_df[sst_columns]

    col_names = num_columns + cat_columns
    sst_col_indices = [col_names.index(name) for name in sst_columns]
    d_num_x = len(num_columns)

    cat_ordinal_mapping = {}
    for col_name, cat in zip(cat_columns, cat_ord_enc.categories_):
        cat_ordinal_mapping[col_name] = list(cat)
    cat_od_y_fn = {}
    for col_name, cat in zip([label_col], label_ord_enc.categories_):
        cat_od_y_fn[col_name] = list(cat)

    n_unq_y = len(cat_od_y_fn[label_col])
    n_unq_sst_lst = [len(cat_ordinal_mapping[name]) for name in sst_columns]
    n_unq_c_lst = [n_unq_y] + n_unq_sst_lst
    n_classes_lst = [len(cate) for cate in cat_ord_enc.categories_]
    return {
        'feature_df': feature_df,
        'label_df': label_df,
        'col_names': col_names,
        'sst_col_indices': sst_col_indices,
        'd_num_x': d_num_x,
        'cat_ordinal_mapping': cat_ordinal_mapping,
        'cat_od_y_fn': cat_od_y_fn,
        'cat_ord_enc': cat_ord_enc,
        'label_ord_enc': label_ord_enc,
        'n_unq_y': n_unq_y,
        'n_unq_c_lst': n_unq_c_lst,
        'n_classes_lst': n_classes_lst,
    }

def get_openml_dataset(data_id: int) -> Dict[str, pd.DataFrame or pd.Series]:
    data_frame = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
    features = data_frame.data
    labels = data_frame.target
    return {'features': features, 'labels': labels}

def save_openml_dataset(name: str, x_norm_type='quantile', ratios=(0.5, 0.5), seed=42, dir_path=None):
    dataset_mapping = {
        'adult': {
            'dataset_name': 'adult', 'data_id': UCI_ADULT_ID,
            'dataset_type': 'tabular', 'n_channels': 1, 'label_col': 'class',
            'cat_columns': [
                'workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'native-country',
            ],
            'sst_columns': ['sex', 'race'],
        },
        'german': {
            'dataset_name': 'german', 'data_id': UCI_GERMAN_CREDIT_ID,
            'dataset_type': 'tabular', 'n_channels': 1, 'label_col': 'Risk',
            'cat_columns': [
                'Sex', 'Job', 'Housing',
                'Saving accounts', 'Checking account', 'Purpose',
            ],
            'sst_columns': ['Sex'],
        },
        'bank': {
            'dataset_name': 'bank', 'data_id': BANK_MARKETING_ID,
            'dataset_type': 'tabular', 'n_channels': 1, 'label_col': 'y',
            'cat_columns': [
                'job', 'marital', 'education', 'default', 'housing', 
                'loan', 'contact', 'month', 'poutcome', 'age-group',
            ],
            'sst_columns': ['age-group'],
        },
        'law': {
            'dataset_name': 'law', 'data_id': LAW_SCHOOL_ID,
            'dataset_type': 'tabular', 'n_channels': 1, 'label_col': 'ugpagt3',
            'cat_columns': ['decile1', 'decile3', 'fam_inc', 'gender', 'race1', 'cluster', 'fulltime', 'bar'],
            'sst_columns':  ['gender', 'race1'],
        },
        'compass': {
            'dataset_name': 'compass', 'data_id': COMPASS_ID,
            'dataset_type': 'tabular', 'n_channels': 1, 'label_col': 'is_recid',
            'cat_columns': [
                'sex', 'age_cat', 'race', 'c_charge_degree', 'decile_score.1',
                'score_text', 'v_type_of_assessment', 'v_decile_score', 'v_score_text',
            ],
            'sst_columns':  ['sex', 'race'],
        },
    }
    data_ref_dict = dataset_mapping[name]
    
    # start of getting dataset
    data_dict = get_openml_dataset(data_ref_dict['data_id'])
    data_features = data_dict['features']
    data_labels = data_dict['labels']
    
    # if dataset is bank, we need to add a column for age-group with a threshold of 25
    if name == 'bank':
        data_features['age-group'] = data_features['age'].apply(lambda x: 'young' if x <= 25 else 'old')
    
    # get necessary intermediate variables
    feature_columns = data_features.columns
    data_all = pd.concat([data_features, data_labels], axis=1)
    data_all = data_all.dropna()
    label_ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    data_all_cp = data_all.copy()
    label_col = data_ref_dict['label_col']
    data_label = data_all[[label_col]].astype(str)
    data_all['temp'] = label_ord_enc.fit_transform(data_label)
    data_all = data_all.drop(columns=[label_col])
    data_all = data_all.rename(columns={'temp': label_col})
    
    # features and labels
    features, labels = data_all[feature_columns], data_all[label_col]
    
    # columns
    cat_columns = data_ref_dict['cat_columns']
    sst_columns = data_ref_dict['sst_columns']
    num_columns = [column for column in features.columns if column not in cat_columns]
    
    # getting information from dataset
    data_ans = get_vars_from_data(
        features=features, labels=labels, sst_columns=sst_columns, cat_columns=cat_columns,
        num_columns=num_columns, label_col=label_col, label_ord_enc=label_ord_enc,
    )
    dataset_name = data_ref_dict['dataset_name']
    dataset_type = data_ref_dict['dataset_type']
    n_channels = data_ref_dict['n_channels']

    # process dataset
    ans = process_dataset(
        dataset_name=dataset_name, dataset_type=dataset_type, n_channels=n_channels,
        data_all=data_all, data_all_cp=data_all_cp, feature_df=data_ans['feature_df'], label_df=data_ans['label_df'],
        x_norm_type=x_norm_type, col_names=data_ans['col_names'], label_col_name=label_col,
        sst_columns=sst_columns, d_num_x=data_ans['d_num_x'],
        cat_columns=cat_columns, num_columns=num_columns, n_classes_lst=data_ans['n_classes_lst'],
        sst_col_indices=data_ans['sst_col_indices'], cat_ordinal_mapping=data_ans['cat_ordinal_mapping'],
        cat_od_y_fn=data_ans['cat_od_y_fn'], cat_ord_enc=data_ans['cat_ord_enc'], label_ord_enc=label_ord_enc,
        n_unq_y=data_ans['n_unq_y'], n_unq_c_lst=data_ans['n_unq_c_lst'],
        ratios=ratios, seed=seed, dir_path=dir_path,
    )
    return ans
