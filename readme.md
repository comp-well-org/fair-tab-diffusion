# Fair Tabular Diffusion

<!-- table of contents -->
## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
  * [Environment](#environment)
- [To do](#to-do)
- [Running](#running)
    * [Experiments](#experiments)
    * [Assessing the quality of synthetic data](#assessing-the-quality-of-synthetic-data)
- [Benchmarks](#benchmarks)
    * [Datasets](#datasets)
    * [Baselines](#baselines)
- [Contact](#contact)

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

## Running

### Experiments

Under the root directory, run the following commands to reproduce the results of our method:

```bash
# download and preprocess datasets
python build.py
# run experiments for our method
bash fairtabddpm.sh
```

To reproduce the results of baseline methods, run the following commands:

```bash
# go to baselines directory
cd baselines
# run experiments for baselines
bash codi.sh
bash fairsmote.sh
bash fairtabgan.sh
bash goggle.sh
bash great.sh
bash smote.sh
bash stasy.sh
bash tabddpm.sh
bash tabsyn.sh
```

### Assessing the quality of synthetic data

TBA

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
