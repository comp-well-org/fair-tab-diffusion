import os
import optuna
import shutil
import warnings
import argparse
import subprocess
import lib
from constant import EXPS_PATH
import numpy as np

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='adult')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    n_trials = args.n_trials
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    base_config_path = f'./args/{dataset}/fairtabddpm/config.toml'
    
    def objective(trial):        
        lr = trial.suggest_float('lr', 0.00001, 0.003, log=True)
        n_epochs = trial.suggest_categorical('n_epochs', [100, 500, 1000])
        n_timesteps = trial.suggest_categorical('n_timesteps', [100, 1000])
        
        base_config = lib.load_config(base_config_path)
        exp_name = 'many-exps'
        
        exp_dir = os.path.join(
            base_config['exp']['home'],
            base_config['data']['name'],
            base_config['exp']['method'],
            exp_name,
        )
        os.makedirs(exp_dir, exist_ok=True)
        
        base_config['train']['lr'] = lr
        base_config['train']['n_epochs'] = n_epochs
        base_config['model']['n_timesteps'] = n_timesteps
        base_config['exp']['device'] = f'cuda:{args.gpu_id}'
        
        trial.set_user_attr('config', base_config)
        lib.write_config(base_config, f'{exp_dir}/config.toml')
        
        subprocess.run(
            [
                'python3.10',
                'fairtabddpm_run.py',
                '--config',
                f'{exp_dir}/config.toml',
                '--exp_name',
                exp_name,
            ],
            check=True,
        )
        report_path = f'{exp_dir}/metric.json'
        report = lib.load_json(report_path)
        score = np.mean(report['CatBoost']['AUC']['Train'])
        
        return score
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_config_dir = os.path.join(EXPS_PATH, dataset, 'fairtabddpm', 'best')
    os.makedirs(best_config_dir, exist_ok=True)
    best_config_path = os.path.join(best_config_dir, 'config.toml')
    
    best_config = study.best_trial.user_attrs['config']
    
    lib.write_config(best_config, best_config_path)
    lib.write_json(optuna.importance.get_param_importances(study), os.path.join(best_config_dir, 'importance.json'))
    
    subprocess.run(
        [
            'python3.10',
            'fairtabddpm_run.py',
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
            'fairtabddpm',
            'many-exps',
        ),
    )

if __name__ == '__main__':
    main()
