import os
from src.datagen.openml import save_openml_dataset
from constant import DB_PATH

def main():
    save_openml_dataset('adult', dir_path=os.path.join(DB_PATH, 'adult'))
    save_openml_dataset('german', dir_path=os.path.join(DB_PATH, 'german'))
    save_openml_dataset('bank', dir_path=os.path.join(DB_PATH, 'bank'))
    save_openml_dataset('law', dir_path=os.path.join(DB_PATH, 'law'))
    save_openml_dataset('compass', dir_path=os.path.join(DB_PATH, 'compass'))

if __name__ == '__main__':
    main()
