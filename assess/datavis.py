import os
import sys
import json
import warnings
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH, PLOTS_PATH
from lib import load_json, load_config

warnings.filterwarnings('ignore')

METHOD_MAPPER = {
    'codi': 'CoDi',
    'fairsmote': 'FairSMOTE',
    'fairtabgan': 'FairTabGAN',
    'goggle': 'Goggle',
    'great': 'GReaT',
    'smote': 'SMOTE',
    'stasy': 'STaSy',
    'tabddpm': 'TabDDPM',
    'tabsyn': 'TabSyn',
    'fairtabddpm': 'Ours',
}

DATASET_MAPPER = {
    'adult': 'Adult',
    'compass': 'COMPAS',
    'german': 'German',
    'bank': 'Bank',
    'law': 'Law',
}

def read_data(data_dir, flag='train'):
    x = pd.read_csv(os.path.join(data_dir, f'x_{flag}.csv'), index_col=0)
    y = pd.read_csv(os.path.join(data_dir, f'y_{flag}.csv'), index_col=0)
    if y.shape[1] > 1:
        y = y.iloc[:, 0]
    data = pd.concat([x, y], axis=1)
    return data

def plot_col_distribution(
    dataset: str,
    config: dict,
    save_path: str = PLOTS_PATH,
):  
    # intialization
    data_dirs = {}
    seed = config['exp']['seed']
    
    # real data
    data_dirs['real'] = os.path.join(DB_PATH, dataset)
    data_desc = load_json(os.path.join(data_dirs['real'], 'desc.json'))
    num_col_names = data_desc['num_col_names']
    cat_col_names = data_desc['cat_col_names'] + [data_desc['label_col_name']]
    print('numerical columns:', num_col_names)
    print('categorical columns:', cat_col_names)
    print()
    
    # have a dictionary to store data for every method
    data_dicts = {}
    data_dicts['real'] = read_data(data_dirs['real'], flag='train')
    
    # synthetic data for every considered method
    considered = config['methods']['considered']
    for method in considered:
        session = config['methods'][method]['session']
        data_dirs[method] = os.path.join(EXPS_PATH, dataset, method, session, 'synthesis', str(seed))
        data_dicts[method] = read_data(data_dirs[method], flag='syn')
    
    for method in data_dicts:
        print(method)
        print(data_dicts[method].head(3))
        print()
    
    # numerical features
    for num_col in num_col_names:
        fig, ax = plt.subplots()
        ax.set_title(DATASET_MAPPER[dataset])
        ax.set_xlabel(num_col)
        ax.set_ylabel('Density')
        
        # save plot
        filename = f'{dataset}_num_{num_col}_dist.png'.strip().lower()
        save_path = os.path.join(PLOTS_PATH, filename)
        plt.savefig(save_path)
        break
    
    # categorical features
    for cat_col in cat_col_names:
        fig, ax = plt.subplots()
        ax.set_xlabel(cat_col)
        ax.set_ylabel('Count')
        
        # save plot
        filename = f'{dataset}_cat_{cat_col}_dist.png'.strip().lower()
        save_path = os.path.join(PLOTS_PATH, filename)
        plt.savefig(save_path)
        break
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--config', type=str, default='./assess.toml')

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # message
    # print(json.dumps(config, indent=4))
    print('-' * 80)
    
    # plot numerical distribution
    plot_col_distribution(
        dataset=args.dataset,
        config=config,
    )

if __name__ == '__main__':
    main()
