import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from datavis import DATASET_MAPPER, METHOD_MAPPER

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH, PLOTS_PATH
from lib import load_json, load_config, write_json

warnings.filterwarnings('ignore')

def read_ord_data(data_dir, data_desc, flag='train'):
    x = pd.read_csv(os.path.join(data_dir, f'x_{flag}.csv'), index_col=0)
    y = pd.read_csv(os.path.join(data_dir, f'y_{flag}.csv'), index_col=0)
    if y.shape[1] > 1:
        y = y.iloc[:, 0]
    data = pd.concat([x, y], axis=1)
    # convert data types
    data = data.astype(data_desc['d_types'])
    return data

TREND_SCORE_COLS = ['Score', 'Real Correlation', 'Synthetic Correlation']
VALIDITY_SCORE_COLS = ['Score']
SHAPE_SCORE_COLS = ['Score']

def eval_density_method_seeds(dataset, config, save_dir):
    # intialization
    data_dirs = {}
    seed = config['exp']['seed']
    n_seeds = config['exp']['n_seeds']

    # real data
    data_dirs['real'] = os.path.join(DB_PATH, dataset)
    data_desc = load_json(os.path.join(data_dirs['real'], 'desc.json'))
    num_col_names = data_desc['num_col_names']
    cat_col_names = data_desc['cat_col_names'] + [data_desc['label_col_name']]
    columns = {}
    for col in num_col_names:
        columns[col] = {
            'sdtype': 'numerical',
        }
    for col in cat_col_names:
        columns[col] = {
            'sdtype': 'categorical',
        }
    metadata = {'columns': columns}
    real_data = read_ord_data(data_dirs['real'], data_desc, flag='train')
    
    # have a dictionary to store data for every method
    score_dict = {}
    table_dict_temp = {}
    table_dict = {}
    
    # synthetic data for every considered method
    considered = config['methods']['considered']
    for method in considered:
        session = config['methods'][method]['session']
        score_dict[method] = {
            'shape': [],
            'trend': [],
            'quality': [],
        }
        table_dict_temp[method] = {
            'shapes': [],
            'trends': [],
            'validities': [],
        }
        table_dict[method] = {
            'shapes': [],
            'trends': [],
            'validities': [],
        }
        for i in range(n_seeds):
            rand_seed = seed + i
            syn_data_path = os.path.join(EXPS_PATH, dataset, method, session, 'synthesis', str(rand_seed))
            syn_data = read_ord_data(syn_data_path, data_desc, flag='syn')
        
            qual_report = QualityReport()
            diag_report = DiagnosticReport()
    
            qual_report.generate(real_data, syn_data, metadata)
            diag_report.generate(real_data, syn_data, metadata)
    
            quality = qual_report.get_properties()
            shape_score = quality['Score'][0]
            trend_score = quality['Score'][1]
            quality_score = (shape_score + trend_score) / 2
            
            score_dict[method]['shape'].append(shape_score)
            score_dict[method]['trend'].append(trend_score)
            score_dict[method]['quality'].append(quality_score)
        
            shapes_table = qual_report.get_details(property_name='Column Shapes')
            trends_table = qual_report.get_details(property_name='Column Pair Trends')
            validities_table = diag_report.get_details('Data Validity')
            
            table_dict_temp[method]['shapes'].append(shapes_table)
            table_dict_temp[method]['trends'].append(trends_table)
            table_dict_temp[method]['validities'].append(validities_table)
        
        # convert score columns in several tables in to a list and go to one table
        trend_scores = []
        shape_scores = []
        validity_scores = []
        for i in range(n_seeds):
            trend_scores.append(table_dict_temp[method]['trends'][i][TREND_SCORE_COLS].values)
            shape_scores.append(table_dict_temp[method]['shapes'][i][SHAPE_SCORE_COLS].values)
            validity_scores.append(table_dict_temp[method]['validities'][i][VALIDITY_SCORE_COLS].values)
            
        combined_trend = [[list(elements) for elements in zip(*rows)] for rows in zip(*trend_scores)]
        combined_shape = [[list(elements) for elements in zip(*rows)] for rows in zip(*shape_scores)]
        combined_validity = [[list(elements) for elements in zip(*rows)] for rows in zip(*validity_scores)]
        
        table_dict[method]['trends'] = table_dict_temp[method]['trends'][0]
        table_dict[method]['shapes'] = table_dict_temp[method]['shapes'][0]
        table_dict[method]['validities'] = table_dict_temp[method]['validities'][0]
        
        table_dict[method]['trends'][TREND_SCORE_COLS] = combined_trend
        table_dict[method]['shapes'][SHAPE_SCORE_COLS] = combined_shape
        table_dict[method]['validities'][VALIDITY_SCORE_COLS] = combined_validity
    
    method = considered[0]
    trends = table_dict[method]['trends']
    shapes = table_dict[method]['shapes']
    validities = table_dict[method]['validities']
    method_name = METHOD_MAPPER[method]
    
    # rename columns
    trends = trends.rename(columns={
        'Score': f'{method_name} - Score',
        'Real Correlation': f'{method_name} - Real Correlation',
        'Synthetic Correlation': f'{method_name} - Synthetic Correlation',
    })
    shapes = shapes.rename(columns={'Score': f'{method_name} - Score'})
    validities = validities.rename(columns={'Score': f'{method_name} - Score'})
    
    # process the rest of the methods
    if len(considered) > 1:
        for method in considered[1:]:
            method_trends = table_dict[method]['trends']
            method_shapes = table_dict[method]['shapes']
            method_validities = table_dict[method]['validities']
            method_name = METHOD_MAPPER[method]
            for col in TREND_SCORE_COLS:
                trends[f'{method_name} - {col}'] = method_trends[col]
            for col in SHAPE_SCORE_COLS:
                shapes[f'{method_name} - {col}'] = method_shapes[col]
            for col in VALIDITY_SCORE_COLS:
                validities[f'{method_name} - {col}'] = method_validities[col]
                
    # write results
    write_json(score_dict, os.path.join(save_dir, 'score.json'))
    trends.to_parquet(os.path.join(save_dir, 'trends.parquet'))
    shapes.to_parquet(os.path.join(save_dir, 'shapes.parquet'))
    validities.to_parquet(os.path.join(save_dir, 'validities.parquet'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--config', type=str, default='./assess.toml')
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')

    # divider
    print('-' * 80)
    
    # save results
    save_dir = f'eval/density/{args.dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    eval_density_method_seeds(args.dataset, config, save_dir)

if __name__ == '__main__':
    main()
