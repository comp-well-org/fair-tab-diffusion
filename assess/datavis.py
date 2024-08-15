import os
import sys
import warnings
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH, PLOTS_PATH
from lib import load_json, load_config, load_encoder

warnings.filterwarnings('ignore')

METHOD_MAPPER = {
    'real': 'Real',
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

def read_data(
    data_dir, cat_col_names, label_col_name, d_types, 
    original=True,
    cat_encoder=None,
    label_encoder=None,
    flag='train',
):
    x = pd.read_csv(os.path.join(data_dir, f'x_{flag}.csv'), index_col=0)
    y = pd.read_csv(os.path.join(data_dir, f'y_{flag}.csv'), index_col=0)
    if y.shape[1] > 1:
        y = y.iloc[:, 0]
    data = pd.concat([x, y], axis=1)
    # convert data types
    data = data.astype(d_types)
    
    if original and cat_encoder and label_encoder:
        # inverse transform for categorical columns
        original_cat_cols = cat_encoder.inverse_transform(data[cat_col_names])
        data[cat_col_names] = original_cat_cols
        
        # inverse transform for label column
        original_label = label_encoder.inverse_transform(data[[label_col_name]])
        data[label_col_name] = original_label.reshape(-1)
    
    return data

def plot_col_distribution(
    dataset: str,
    config: dict,
    save_path: str = PLOTS_PATH,
    num_plot: bool = True,
    cat_plot: bool = True,
):  
    # intialization
    data_dirs = {}
    seed = config['exp']['seed']
    
    # save path
    plot_dir = os.path.join(save_path, dataset)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # real data
    data_dirs['real'] = os.path.join(DB_PATH, dataset)
    data_desc = load_json(os.path.join(data_dirs['real'], 'desc.json'))
    num_col_names = data_desc['num_col_names']
    cat_col_names = data_desc['cat_col_names']
    label_col_name = data_desc['label_col_name']
    cat_encoder = load_encoder(os.path.join(data_dirs['real'], 'cat_encoder.skops'))
    label_encoder = load_encoder(os.path.join(data_dirs['real'], 'label_encoder.skops'))
    d_types = data_desc['d_types']
    all_cat_col_names = cat_col_names + [label_col_name]
    
    # have a dictionary to store data for every method
    data_dicts = {}
    data_dicts['real'] = read_data(
        data_dirs['real'], cat_col_names, label_col_name, d_types,
        original=True, cat_encoder=cat_encoder, label_encoder=label_encoder,
        flag='test',
    )
    
    # synthetic data for every considered method
    considered = config['methods']['considered']
    for method in considered:
        session = config['methods'][method]['session']
        data_dirs[method] = os.path.join(EXPS_PATH, dataset, method, session, 'synthesis', str(seed))
        data_dicts[method] = read_data(
            data_dirs[method], cat_col_names, label_col_name, d_types, 
            original=True, cat_encoder=cat_encoder, label_encoder=label_encoder,
            flag='syn',
        )
    
    # take a look at the data
    print_df = False
    # print_df = True
    if print_df:
        for method in data_dicts:
            print(method)
            print(data_dicts[method].head(3))
            print()
    
    if num_plot:
        # numerical features
        for num_col in num_col_names:
            # initialize plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(DATASET_MAPPER[dataset])
            ax.set_xlabel(num_col)
            ax.set_ylabel('Density')
            plt.tight_layout()
            
            # plot data
            for method in data_dicts:
                data = data_dicts[method]
                sns.kdeplot(data[num_col], ax=ax, label=METHOD_MAPPER[method], shade=True, linewidth=0.5)
            
            # yticklabels
            ax.ticklabel_format(axis='y', scilimits=[-1, 1], style='sci')
            ax.legend()
            fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
            # set n yticks to be 6
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            
            # save plot
            filename = f'num_{num_col}_dist.pdf'.strip().lower()
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path)
    
    if cat_plot:
        # categorical features
        for cat_col in all_cat_col_names:
            # initialize plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel(cat_col)
            ax.set_ylabel('Count')
            plt.tight_layout()
            
            # plot data with bar plot
            cat_count_dict = {}
            for method in data_dicts:
                data = data_dicts[method]
                cat_count = data[cat_col].value_counts().to_dict()
                cat_count_dict[METHOD_MAPPER[method]] = cat_count
            cat_count_df = pd.DataFrame(cat_count_dict)
            cat_count_df.plot(kind='bar', ax=ax, cmap='tab20c')
            
            # yticklabels
            ax.ticklabel_format(axis='y', scilimits=[-1, 1], style='sci')
            ax.legend()
            fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10)
            # set n yticks to be 6
            ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            # rotate xticklabels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            # save plot
            filename = f'cat_{cat_col}_dist.pdf'.strip().lower()
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--config', type=str, default='./assess.toml')
    parser.add_argument('--plot_dist', action='store_true', default=True)

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # divider
    print('-' * 80)
    
    if args.plot_dist:
        # plot numerical distribution
        plot_col_distribution(
            dataset=args.dataset,
            config=config,
        )
        print(f'plots are saved in {PLOTS_PATH}/{args.dataset}')

if __name__ == '__main__':
    main()
