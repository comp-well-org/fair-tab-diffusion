import os
import time
import json
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import skops.io as sio
from src.diffusion.utils import XYCTabDataModule
from src.diffusion.estimator import PosteriorEstimator, DenoiseFn
from src.diffusion.configs import DenoiseFnCfg, DataCfg, GuidCfg
from src.diffusion.unet import Unet
from src.diffusion.ddpm import GaussianMultinomialDiffusion
from src.diffusion.trainer import XYCTabTrainer
from src.evaluate.skmodels import default_sk_clf
from lib import load_config, copy_file
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    parser.add_argument('--train', action='store_true', help='training', default=True)
    parser.add_argument('--sample', action='store_true', help='sampling', default=True)
    parser.add_argument('--eval', action='store_true', help='evaluation', default=True)
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # configs
    exp_config = config['exp']
    guid_config = config['guid']
    data_config = config['data']
    model_config = config['model']
    sample_config = config['sample']
    eval_config = config['eval']
    
    # number of random seeds for sampling
    n_seeds = sample_config['n_seeds']
    
    # message
    print(json.dumps(config, indent=4))
    print('-' * 80)
    
    # whether to use fair diffusion model
    is_fair = exp_config['fair']
    
    # random seed
    seed = exp_config['seed']
    torch.manual_seed(seed)
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        args.exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        args.config,
    )
    
    # data
    data_module = XYCTabDataModule(
        root=os.path.join(data_config['path'], data_config['name']),
        batch_size=data_config['batch_size'],
    )
    data_desc = data_module.get_data_description()
    norm_fn = data_module.get_norm_fn()
    empirical_dist = data_module.get_empirical_dist()

    # arguments determined by data
    d_oh_x = data_desc['d_oh_x']
    d_num_x = data_desc['d_num_x']
    n_channels = data_desc['n_channels']
    n_unq_c_lst = data_desc['n_unq_c_lst']
    n_unq_cat_od_x_lst = data_desc['n_unq_cat_od_x_lst']
    
    # fairness
    is_fair = exp_config['fair']
    cond_emb_factor = 3 if is_fair else 2
    
    # denoising function
    denoise_fn = DenoiseFn(
        denoise_fn_cfg=DenoiseFnCfg(
            d_x_emb=model_config['d_x_emb'],
            d_t_emb=model_config['d_t_emb'],
            d_cond_emb=model_config['d_cond_emb'],
        ),
        data_cfg=DataCfg(
            d_oh_x=d_oh_x,
            n_channels=n_channels,
            n_unq_c_lst=n_unq_c_lst,
        ),
        guid_cfg=GuidCfg(
            cond_guid_weight=guid_config['cond_guid_weight'],
            cond_guid_threshold=guid_config['cond_guid_threshold'],
            cond_momentum_weight=guid_config['cond_momentum_weight'],
            cond_momentum_beta=guid_config['cond_momentum_beta'],
            warmup_steps=guid_config['warmup_steps'],
            overall_guid_weight=guid_config['overall_guid_weight'],
        ),
        posterior_est=PosteriorEstimator(
            Unet(
                n_in_channels=n_channels,
                n_out_channels=n_channels,
                n_base_channels=model_config['n_base_channels'],
                n_channels_factors=model_config['n_channels_factors'],
                n_res_blocks=model_config['n_res_blocks'],
                attention_levels=model_config['attention_levels'],
                d_t_emb=model_config['d_t_emb'],
                d_cond_emb=model_config['d_cond_emb'] * cond_emb_factor,
                n_groups=model_config['n_groups'],
                n_heads=model_config['n_heads'],
                n_tf_layers=model_config['n_tf_layers'],
            ),
        ),
    )
    
    # diffusion
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(n_unq_cat_od_x_lst),
        num_numerical_features=d_num_x,
        denoise_fn=denoise_fn,
        device=exp_config['device'],
        scheduler=model_config['scheduler'],
        max_beta=model_config['max_beta'],
        num_timesteps=model_config['n_timesteps'],
        is_fair=is_fair,
        gaussian_parametrization=model_config['parametrization'],
    )
    
    if args.train:
        # train
        train_config = config['train']
        trainer = XYCTabTrainer(
            n_epochs=train_config['n_epochs'],
            lr=train_config['lr'],
            weight_decay=train_config['weight_decay'],
            max_non_improve=train_config['max_non_improve'],
            is_fair=is_fair,
            device=exp_config['device'],
        )
        start_time = time.time()
        trainer.fit(diffusion, data_module, exp_dir)
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
            time_msg = f'time: {end_time - start_time:.2f} sec'
            f.write(time_msg)
    
    if args.sample:
        # load model
        diffusion.load_state_dict(torch.load(os.path.join(exp_dir, 'diffusion.pt')))
        diffusion.to(exp_config['device'])
        diffusion.eval()
        assert n_seeds > 0, '`n_seeds` must be greater than 0'
        
        # distribution of conditionals
        assert sample_config['dist'] in {'uniform', 'empirical', 'fair'}
        if sample_config['dist'] == 'uniform':
            cond_dist = [torch.ones(n) for n in n_unq_c_lst]
        elif sample_config['dist'] == 'empirical':
            cond_dist = empirical_dist
        elif sample_config['dist'] == 'fair':
            cond_dist = [torch.ones(n) for n in n_unq_c_lst]
            cond_dist[0] = empirical_dist[0]
        
        feature_cols, label_cols = data_module.get_feature_label_cols()
        
        # sampling with seeds
        for i in range(n_seeds):
            random_seed = seed + i
            torch.manual_seed(random_seed)
            xn, cond = diffusion.sample_all(
                sample_config['n_samples'],
                cond_dist,
                batch_size=sample_config['batch_size'],
            )
            print(f'synthetic data shape: {list(xn.shape)}, synthetic cond shape: {list(cond.shape)}')
            
            # inverse normalization
            x_num = norm_fn.inverse_transform(xn[:, :d_num_x])
            x_syn = np.concatenate([x_num, xn[:, d_num_x:]], axis=1)
            x_syn = pd.DataFrame(x_syn, columns=feature_cols)
            xn_syn = pd.DataFrame(xn, columns=feature_cols)
            y_syn = pd.DataFrame(cond, columns=label_cols)
            
            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)
            x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            xn_syn.to_csv(os.path.join(synth_dir, 'xn_syn.csv'))
            y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
        
        # copy `data_desc` as json file and `norm_fn` as skops file
        synth_dir = os.path.join(exp_dir, 'synthesis')
        with open(os.path.join(synth_dir, 'desc.json'), 'w') as f:
            json.dump(data_desc, f, indent=4)
        sio.dump(norm_fn, os.path.join(synth_dir, 'fn.skops'))
    
    if args.eval:
        # read validation data
        x_eval = pd.read_csv(
            os.path.join(data_config['path'], data_config['name'], 'x_eval.csv'),
            index_col=0,
        )
        c_eval = pd.read_csv(
            os.path.join(data_config['path'], data_config['name'], 'y_eval.csv'),
            index_col=0,
        )
        y_eval = c_eval.iloc[:, 0]
        
        # evaluate classifiers trained on synthetic data
        metric = {}
        for clf_choice in eval_config['sk_clf_choice']:
            aucs = []
            for i in range(n_seeds):
                random_seed = seed + i
                synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
                
                # read synthetic data
                x_syn = pd.read_csv(os.path.join(synth_dir, 'x_syn.csv'), index_col=0)
                c_syn = pd.read_csv(os.path.join(synth_dir, 'y_syn.csv'), index_col=0)
                y_syn = c_syn.iloc[:, 0]
                
                # train classifier
                clf = default_sk_clf(clf_choice, random_seed)
                clf.fit(x_syn, y_syn)
                y_pred = clf.predict_proba(x_eval)[:, 1]
                aucs.append(roc_auc_score(y_eval, y_pred))
            metric[clf_choice] = (np.mean(aucs), np.std(aucs))
        with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
            json.dump(metric, f, indent=4)

if __name__ == '__main__':
    main()
