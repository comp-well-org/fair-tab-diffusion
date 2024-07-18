from src.data.openml import save_adult
from constant import ADULT_PATH

def main():
    save_adult(x_norm_type='quantile', dir_path=ADULT_PATH)

if __name__ == '__main__':
    main()
