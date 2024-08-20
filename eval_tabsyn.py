import os
import sys
import time
import math
import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import skops.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import load_config, copy_file, load_json
from src.evaluate.metrics import evaluate_syn_data, print_metric
from constant import DB_PATH, EXPS_PATH

def main():
    # configs
    dataset = 'german'
    exp_name = 'best'
    config_path = os.path.join(EXPS_PATH, dataset, 'tabsyn', exp_name, 'config.toml')
    
    config = load_config(config_path)
    exp_config = config['exp']
    data_config = config['data']
    sample_config = config['sample']
    eval_config = config['eval']
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        config,
    )
    
    seed = exp_config['seed']
    n_seeds = sample_config['n_seeds']
    
    # data
    dataset_dir = os.path.join(data_config['path'], data_config['name'])
    data_desc = load_json(os.path.join(dataset_dir, 'desc.json'))
    cat_encoder = sio.load(os.path.join(dataset_dir, 'cat_encoder.skops'))
    label_encoder = sio.load(os.path.join(dataset_dir, 'label_encoder.skops'))
    
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
