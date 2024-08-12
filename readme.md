# Fair Tabular Diffusion

## Setup

### Environment

The PyTorch version we used in this project is `2.3.0+cu121`, and you can install the required packages by running the following command:

```bash
conda create -n ai python=3.10
source activate ai
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## To do

Avoid repeatition to improve the code quality:

- [ ] Replace `exp_config['home']` by importing `EXPS_PATH` from `constant.py` in all running scripts
- [ ] Replace `data_config['path']` by importing `DB_PATH` from `constant.py` in all running scripts
- [ ] Delete home of experiments and path of datasets in all `config.toml` files
- [ ] Add a new argument `--method` to optimization scripts and merge all optimization scripts into one
- [ ] Find commonly used functions in all running scripts and move them to `utils.py`

## Benchmarks

### Datasets

- Adult
- COMPASS
- German Credit
- Bank Marketing
- Law School Admissions

### Baselines

The baseline methods we used in this project are as follows (sorted alphabetically):

- [X] CoDi
- [X] Fair SMOTE
- [X] Fair Tabular GAN
- [X] Goggle
- [X] Great
- [X] SMOTE
- [X] STaSy
- [X] TabDDPM
- [X] TabSyn

## Contact

- Zeyu Yang
