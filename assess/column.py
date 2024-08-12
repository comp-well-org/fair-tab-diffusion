import os
import sys
import json
import warnings
import argparse
from datavis import read_data
import matplotlib.pyplot as plt

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH, PLOTS_PATH
from lib import load_json, load_config

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./assess.toml')

    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # message
    # print(json.dumps(config, indent=4))
    print('-' * 80)

if __name__ == '__main__':
    main()
    # low-order stats: column-wise density estimation and pair-wise column correlation
    # high-order stats: alpha-precision and beta-recall scores that measure the overall fidelity and diversity of synthetic data
