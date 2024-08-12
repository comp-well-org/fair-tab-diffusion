import os
import sys
import warnings
import argparse

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

from constant import DB_PATH, EXPS_PATH
from lib import load_json

warnings.filterwarnings('ignore')

def main():
    pass
    # low-order stats: column-wise density estimation and pair-wise column correlation
    # high-order stats: alpha-precision and beta-recall scores that measure the overall fidelity and diversity of synthetic data

if __name__ == '__main__':
    main()
