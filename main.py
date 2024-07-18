import argparse
import tomli
from src.diffusion.utils import XYCTabDataModule
from src.diffusion.estimator import PosteriorEstimator, DenoiseFn

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


if __name__ == '__main__':
    main()
