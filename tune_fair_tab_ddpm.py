import os
import optuna
import shutil
import warnings
import argparse
import subprocess
import lib
from constant import EXPS_PATH

warnings.filterwarnings('ignore')

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
    base_config_path = f'./args/{dataset}/fair-tab-ddpm/config.toml'
    
    def objective(trial):        
        lr = trial.suggest_float('lr', 0.00001, 0.003, log=True)
        n_epochs = trial.suggest_categorical('n_epochs', [500, 4000])
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
        
        trial.set_user_attr('config', base_config)
        lib.write_config(base_config, f'{exp_dir}/config.toml')
        
        subprocess.run(
            [
                'python3.10',
                'pipeline.py',
                '--config',
                f'{exp_dir}/config.toml',
                '--exp_name',
                exp_name,
                '--train',
                '--sample',
                '--eval',
                '--override',
            ],
            check=True,
        )
        report_path = f'{exp_dir}/metric.json'
        report = lib.load_json(report_path)
        score = report['CatBoost'][0]
        
        return score
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_config_dir = os.path.join(EXPS_PATH, dataset, 'fair-tab-ddpm', 'best')
    os.makedirs(best_config_dir, exist_ok=True)
    best_config_path = os.path.join(best_config_dir, 'config.toml')
    
    best_config = study.best_trial.user_attrs['config']
    
    lib.write_config(best_config, best_config_path)
    lib.write_json(optuna.importance.get_param_importances(study), os.path.join(best_config_dir, 'importance.json'))
    
    subprocess.run(
        [
            'python3.10',
            'pipeline.py',
            '--exp_name',
            'best',
            '--config',
            f'{best_config_path}',
            '--train',
            '--sample',
            '--eval',
            '--override',
        ],
        check=True,
    )
    shutil.rmtree(
        os.path.join(
            EXPS_PATH,
            dataset,
            'fair-tab-ddpm',
            'many-exps',
        ),
    )

if __name__ == '__main__':
    main()
