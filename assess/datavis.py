import os
import sys
import warnings
import argparse
import numpy as np
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
    'fairsmote': 'FairCB',
    'fairtabgan': 'FairTGAN',
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

def clean_nested_dict(report: dict):
    # strip and make title the keys
    clean_report = {}
    for key, value in report.items():
        key = key.strip().title()
        clean_report[key] = {}
        for k, v in value.items():
            k = k.replace("'", '')
            k = k.strip().title()
            clean_report[key][k] = v
    return clean_report

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

def plot_attribute_dist(
    real: dict, synthetic: dict, title: str = 'Real vs Synthetic', path: str = None,
):
    real = clean_nested_dict(real)
    synthetic = clean_nested_dict(synthetic)
    keys_list = []
    for _, value in real.items():
        keys_list += list(value.keys())
    # print(keys_list)
    
    # if there are duplicate keys, then add a suffix to the keys
    if len(keys_list) != len(set(keys_list)):
        real_cp = real.copy()
        synthetic_cp = synthetic.copy()

        # add suffix to the keys of the sub-dictionary
        for suffix in real_cp:
            value = real_cp[suffix]
            new_value = {}
            for k, v in value.items():
                new_value[f'{suffix} {k}'] = v
            real[suffix] = new_value
        
            value = synthetic_cp[suffix]
            new_value = {}
            for k, v in value.items():
                new_value[f'{suffix} {k}'] = v
            synthetic[suffix] = new_value
    #     # rename the keys
    #     for key in real:
    #         value = real[key]
    #         new_value = {}
    #         for k, v in value.items():
    #             new_value[f'{k} (Real)'] = v
    #         real[key] = new_value
        
    # initialize figure and subplots
    # n_attributes = len(real)  
    # if n_attributes == 1:
    #     figsize = (10, 8)
    # elif n_attributes == 2:
    #     figsize = (16, 8)
    # elif n_attributes == 3:
    #     figsize = (16, 8)
    n_unq_total = sum([len(real[key]) for key in real]) 
    figsize = (n_unq_total * 2, 8)
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axs[0].set_title('Real', loc='right', color='blue', fontsize=14)
    axs[1].set_title('Synthetic', loc='right', color='red', fontsize=14)

    # plot real distribution
    real_dfs = []
    for category, data in real.items():
        df = pd.DataFrame(data.items(), columns=['item', 'Percentage'])
        df['category'] = category
        real_dfs.append(df)
    real_all = pd.concat(real_dfs, ignore_index=True)
    ax0 = sns.barplot(data=real_all, x='item', y='Percentage', hue='category', ax=axs[0], legend=True)
    # show percentage on top of the bars
    for p in ax0.containers:
        ax0.bar_label(p, label_type='edge', fontsize=12, fmt='%.4f')
    
    # plot synthetic distribution
    synthetic_dfs = []
    for category, data in synthetic.items():
        df = pd.DataFrame(data.items(), columns=['item', 'Percentage'])
        df['category'] = category
        synthetic_dfs.append(df)
    synthetic_all = pd.concat(synthetic_dfs, ignore_index=True)
    ax1 = sns.barplot(
        data=synthetic_all, x='item', y='Percentage', hue='category', ax=axs[1], legend=False,
    )
    # show percentage on top of the bars
    for p in ax1.containers:
        ax1.bar_label(p, label_type='edge', fontsize=12, fmt='%.4f')

    # rotate x labels
    x = np.arange(len(real_all['item']))
    axs[0].set_xticks(x)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=0)

    # add legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.96), fontsize=12)

    # remove legned from the first subplot and x labels from the second
    axs[0].get_legend().remove()
    axs[1].set_xlabel('')
    axs[0].set_xlabel('')   

    # remove spines
    for ax in axs:
        for side in ('right', 'top'):
            sp = ax.spines[side]
            sp.set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # set y
        ax.set_ylabel('Percentage', fontsize=12)
        
    fig.suptitle(title, y=0.98, fontsize=16)
    plt.tight_layout()
    
    # save figure
    file_name = title.replace(' ', '_').lower()
    if path is not None:
        fig.savefig(os.path.join(path, f'{file_name}.pdf'))
    
    return fig

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
    sst_col_names = data_desc['sst_col_names']
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
                cat_count = data[cat_col].value_counts(normalize=True).to_dict()
                cat_count_dict[METHOD_MAPPER[method]] = cat_count
            cat_count_df = pd.DataFrame(cat_count_dict)
            cat_count_df.plot(kind='bar', ax=ax, cmap='tab20c')
            
            # yticklabels
            # ax.ticklabel_format(axis='y', scilimits=[-1, 1], style='sci')
            ax.legend()
            fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.10)
            # # set n yticks to be 6
            # ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            # rotate xticklabels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            
            ax.set_ylabel('Percentage')
            
            # if number of xticks is more than 6, then rename them as capital letters starting from A
            if len(cat_count_df) > 6:
                ax.set_xticklabels([chr(65 + i) for i in range(len(cat_count_df))])
            
            # save plot
            filename = f'cat_{cat_col}_dist.pdf'.strip().lower()
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path)
            plt.close()
        
    # plot sensitive attributes in the real and synthetic data
    real_sens = {}
    synthetic_sens = {}
    for s_col in sst_col_names:
        real_sens[s_col] = data_dicts['real'][s_col].value_counts(normalize=True).to_dict()
        synthetic_sens[s_col] = data_dicts['fairtabddpm'][s_col].value_counts(normalize=True).to_dict()
    
    # print(real_sens)
    # print(synthetic_sens)
    
    # plot attribute distribution
    plot_attribute_dist(
        real=real_sens, synthetic=synthetic_sens, title=f'Senstive Feature Imbalance in {DATASET_MAPPER[dataset]}',
        path=plot_dir,
    )

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
            save_path=PLOTS_PATH,
            num_plot=False,
            cat_plot=False,
        )
        print(f'plots are saved in {PLOTS_PATH}/{args.dataset}')

if __name__ == '__main__':
    main()
