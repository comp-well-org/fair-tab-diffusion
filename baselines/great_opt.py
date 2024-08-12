import os
import optuna
import shutil
import warnings
import argparse
import subprocess
import sys
import pandas as pd

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

import lib
from constant import EXPS_PATH, ARGS_DIR

warnings.filterwarnings('ignore')

# NOTE: change the method name
METHOD = 'great'
REF_NUM = 4 * (10 ** 4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    n_trials = args.n_trials
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    base_config_path = os.path.join(ARGS_DIR, dataset, f'{METHOD}', 'config.toml')
    
    epoch_list = [5, 10, 20]
    config = lib.load_config(base_config_path)
    data_dir = os.path.join(config['data']['path'], config['data']['name'])
    train_data = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    train_size = train_data.shape[0]
    scale_factor = REF_NUM / train_size
    n_epochs_list = [int(epoch * scale_factor) for epoch in epoch_list]
    
    def objective(trial):
        # NOTE: hyperparameters start here
        batch_size = trial.suggest_categorical('batch_size', [4, 8])
        n_epochs = trial.suggest_categorical('n_epochs', n_epochs_list)
        
        base_config = lib.load_config(base_config_path)
        exp_name = 'many-exps'
        
        exp_dir = os.path.join(
            base_config['exp']['home'],
            base_config['data']['name'],
            base_config['exp']['method'],
            exp_name,
        )
        os.makedirs(exp_dir, exist_ok=True)
        
        # NOTE: edit the config here
        base_config['data']['batch_size'] = batch_size
        base_config['train']['n_epochs'] = n_epochs
        base_config['exp']['device'] = f'cuda:{args.gpu_id}'
        
        trial.set_user_attr('config', base_config)
        lib.write_config(base_config, f'{exp_dir}/config.toml')
        
        subprocess.run(
            [
                'python3.10',
                f'{METHOD}_run.py',
                '--config',
                f'{exp_dir}/config.toml',
                '--exp_name',
                exp_name,
            ],
            check=True,
        )
        report_path = f'{exp_dir}/metric.json'
        report = lib.load_json(report_path)
        val_auc_list = report['CatBoost']['AUC']['Validation']
        score = sum(val_auc_list) / len(val_auc_list)
        
        return score
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_config_dir = os.path.join(EXPS_PATH, dataset, f'{METHOD}', 'best')
    os.makedirs(best_config_dir, exist_ok=True)
    best_config_path = os.path.join(best_config_dir, 'config.toml')
    
    best_config = study.best_trial.user_attrs['config']
    
    lib.write_config(best_config, best_config_path)
    lib.write_json(optuna.importance.get_param_importances(study), os.path.join(best_config_dir, 'importance.json'))
    
    subprocess.run(
        [
            'python3.10',
            f'{METHOD}_run.py',
            '--exp_name',
            'best',
            '--config',
            f'{best_config_path}',
        ],
        check=True,
    )
    shutil.rmtree(
        os.path.join(
            EXPS_PATH,
            dataset,
            f'{METHOD}',
            'many-exps',
        ),
    )

if __name__ == '__main__':
    main()
