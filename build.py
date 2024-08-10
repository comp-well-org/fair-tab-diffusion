import os
from src.datagen.openml import save_adult, save_german_credit
from constant import DB_PATH

def main():
    save_adult(x_norm_type='quantile', dir_path=os.path.join(DB_PATH, 'adult'))
    save_german_credit(x_norm_type='quantile', dir_path=os.path.join(DB_PATH, 'german'))

if __name__ == '__main__':
    main()
