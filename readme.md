# Fair Tabular Diffusion

## Setup

The PyTorch version we used in this project is `2.3.0+cu121`, and you can install the required packages by running the following command:

```bash
conda create -n ai python=3.10
source activate ai
pip install -r requirements.txt
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

To download and preprocess the datasets, run the following command:

```bash
python build.py
```

## Running experiments

Under the root directory, run the following commands to reproduce the results of our method:

```bash
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

## Assessing the quality of synthetic data

TBA

## Benchmarks

### Datasets

- Adult
- COMPASS
- German Credit
- Bank Marketing

### Baselines

The baseline methods we used in this project are as follows (sorted alphabetically):

- [X] CoDi
- [X] Goggle
- [X] GReaT
- [X] SMOTE
- [X] STaSy
- [X] TabDDPM
- [X] TabSyn
- [X] Fair Class Balancing (FCB)
- [X] TabFairGAN

## Contact

- Zeyu Yang

## To do

Avoid repeatition to improve the code quality:

- [ ] Replace `exp_config['home']` by importing `EXPS_PATH` from `constant.py` in all running scripts
- [ ] Replace `data_config['path']` by importing `DB_PATH` from `constant.py` in all running scripts
- [ ] Delete home of experiments and path of datasets in all `config.toml` files
- [ ] Add a new argument `--method` to optimization scripts and merge all optimization scripts into one
- [ ] Find commonly used functions in all running scripts and move them to `utils.py`

Organize the code:

- [ ] Move `fairtabddpm.sh`, `fairtabddpm_run.py`, `fairtabddpm_opt.py` to `baseline` directory and rename `baseline` directory to `methods`, and edit `readme.md` accordingly
- [ ] Move `src/evaluate/metrics.py` out to the root directory because it is specific to the project
