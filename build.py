import os
from src.datagen.openml import save_openml_dataset
from constant import DB_PATH

def main():
    save_openml_dataset('adult', dir_path=os.path.join(DB_PATH, 'adult'))
    save_openml_dataset('german', dir_path=os.path.join(DB_PATH, 'german'))

if __name__ == '__main__':
    main()
