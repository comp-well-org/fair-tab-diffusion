import os
import optuna
import shutil
import warnings
import argparse
import subprocess
import sys

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

import lib
from constant import EXPS_PATH, ARGS_DIR

warnings.filterwarnings('ignore')

METHOD = 'fair-smote'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--n_trials', type=int, default=20)
    args = parser.parse_args()
    dataset = args.dataset
    n_trials = args.n_trials
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    base_config_path = os.path.join(ARGS_DIR, dataset, f'{METHOD}', 'config.toml')
    
    method_str_py = ''.join(METHOD.split('-'))
    
    def objective(trial):        
        knn = trial.suggest_categorical('knn', [2, 21])
        
        base_config = lib.load_config(base_config_path)
        exp_name = 'many-exps'
        
        exp_dir = os.path.join(
            base_config['exp']['home'],
            base_config['data']['name'],
            base_config['exp']['method'],
            exp_name,
        )
        os.makedirs(exp_dir, exist_ok=True)
        
        base_config['model']['knn'] = knn
        
        trial.set_user_attr('config', base_config)
        lib.write_config(base_config, f'{exp_dir}/config.toml')
        print(f'writing config to {exp_dir}/config.toml')
        
        subprocess.run(
            [
                'python3.10',
                f'{method_str_py}_run.py',
                '--config',
                f'{exp_dir}/config.toml',
                '--exp_name',
                exp_name,
            ],
            check=True,
        )
        report_path = f'{exp_dir}/metric.json'
        report = lib.load_json(report_path)
        score = report['CatBoost'][0]
        
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
            f'{method_str_py}_run.py',
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
