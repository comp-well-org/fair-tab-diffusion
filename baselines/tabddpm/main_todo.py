import warnings
import argparse

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    parser.add_argument('--train', action='store_true', help='training')
    parser.add_argument('--sample', action='store_true', help='sampling')
    parser.add_argument('--eval', action='store_true', help='evaluation')
    parser.add_argument('--override', action='store_true', help='override existing model')

if __name__ == '__main__':
    main()
