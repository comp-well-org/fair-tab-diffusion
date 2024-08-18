import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datavis import read_data, clean_nested_dict, DATASET_MAPPER

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH, PLOTS_PATH
from lib import load_json, load_config, load_encoder

warnings.filterwarnings('ignore')

def plot_fair_dist_patches(
    datasets: list[str],
    config: dict,
    save_path: str = PLOTS_PATH,
):  
    # intialization
    data_dirs = {}
    seed = config['exp']['seed']
    
    # save path
    plot_dir = os.path.join(save_path)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # combine all the figures by creating a new figure
    fig = plt.figure(layout='constrained', figsize=(30, 10))
    subfigs = fig.subfigures(1, len(datasets), wspace=0.05)
    
    for dataset in datasets:
        # real data
        data_dirs['real'] = os.path.join(DB_PATH, dataset)
        data_desc = load_json(os.path.join(data_dirs['real'], 'desc.json'))
        cat_col_names = data_desc['cat_col_names']
        sst_col_names = data_desc['sst_col_names']
        label_col_name = data_desc['label_col_name']
        cat_encoder = load_encoder(os.path.join(data_dirs['real'], 'cat_encoder.skops'))
        label_encoder = load_encoder(os.path.join(data_dirs['real'], 'label_encoder.skops'))
        d_types = data_desc['d_types']
        
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
        # plot sensitive attributes in the real and synthetic data
        real_sens = {}
        synthetic_sens = {}
        for s_col in sst_col_names:
            real_sens[s_col] = data_dicts['real'][s_col].value_counts(normalize=True).to_dict()
            synthetic_sens[s_col] = data_dicts['fairtabddpm'][s_col].value_counts(normalize=True).to_dict()
        
        title = DATASET_MAPPER[dataset]
        print(f'Plotting {title}...')
        
        real = clean_nested_dict(real_sens)
        synthetic = clean_nested_dict(synthetic_sens)
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
                        
        axs = subfigs[datasets.index(dataset)].subplots(2, 1, sharex=True)
        axs[0].set_title('Real', loc='right', color='blue', fontsize=6 + 14)
        axs[1].set_title('Synthetic', loc='right', color='red', fontsize=6 + 14)
        
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
            ax0.bar_label(p, label_type='edge', fontsize=6 + 12, fmt='%.4f')
    
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
            ax1.bar_label(p, label_type='edge', fontsize=6 + 12, fmt='%.4f')

        # rotate x labels
        x = np.arange(len(real_all['item']))
        axs[0].set_xticks(x)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=15)
        
        # add legend
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles, labels, fontsize=6 + 12)
        
        # remove legned from the first subplot and x labels from the second
        axs[1].set_xlabel('')
        axs[0].set_xlabel('')   
    
        # remove spines
        for ax in axs:
            for side in ('right', 'top'):
                sp = ax.spines[side]
                sp.set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=2 + 12)
            # set y
            ax.set_ylabel('Percentage', fontsize=6 + 12)

        # add title as the name of the dataset
        axs[0].set_title(title, fontsize=6 + 14)
    
    # add a common title
    fig.suptitle('Sensitive Attribute Distribution in Real and Synthetic Data', fontsize=6 + 16)
    
    # save figure
    file_name = 'real_vs_synthetic_sens_attr_dist'
    fig.savefig(os.path.join(plot_dir, f'{file_name}.pdf'))

if __name__ == '__main__':
    # load the configuration file
    config = load_config('./assess.toml')
    
    # get the datasets
    datasets = ['adult', 'bank', 'compass']
    
    # plot the fair distribution
    plot_fair_dist_patches(datasets, config)
