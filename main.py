import argparse
import tomli
import numpy as np
from src.diffusion.utils import XYCTabDataModule
from src.diffusion.estimator import PosteriorEstimator, DenoiseFn
from src.diffusion.configs import DenoiseFnCfg, DataCfg, GuidCfg
from src.diffusion.unet import Unet
from src.diffusion.ddpm import GaussianMultinomialDiffusion
from src.diffusion.trainer import XYCTabTrainer

def load_config(path) -> dict:
    with open(path, 'rb') as f:
        return tomli.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    
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
    
    data_module = XYCTabDataModule(
        root='/rdf/db/public-tabular-datasets/adult/',
        batch_size=32,
    )

    data_desc = data_module.get_data_description()
    norm_fn = data_module.get_norm_fn()
    empirical_dist = data_module.get_empirical_dist()

    d_oh_x = data_desc['d_oh_x']
    d_num_x = data_desc['d_num_x']
    n_channels = data_desc['n_channels']
    n_unq_c_lst = data_desc['n_unq_c_lst']
    n_unq_cat_od_x_lst = data_desc['n_unq_cat_od_x_lst']
    
    cond_emb_factor = 2
    
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
    
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array(n_unq_cat_od_x_lst),
        num_numerical_features=d_num_x,
        denoise_fn=denoise_fn,
        device=exp_config['device'],
        scheduler=model_config['scheduler'],
        max_beta=model_config['max_beta'],
        num_timesteps=model_config['n_timesteps'],
        is_fair=False,
        gaussian_parametrization=model_config['parametrization'],
    )
    
    # train ^_^
    train_config = config['train']
    trainer = XYCTabTrainer(
        n_epochs=train_config['n_epochs'],
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        max_non_improve=train_config['max_non_improve'],
        is_fair=False,
        device=exp_config['device'],
    )
    trainer.fit(diffusion, data_module, './fair-tab-diffusion-exps')

if __name__ == '__main__':
    main()
