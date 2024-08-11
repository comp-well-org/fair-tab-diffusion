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

## Benchmarks

### Datasets

- Adult
- COMPASS
- German Credit
- Bank Marketing
- Law School Admissions

### Baselines

- [X] CoDi
- [X] Fair SMOTE
- [X] Fair Tabular GAN
- [X] Goggle
- [X] Great
- [] SMOTE
- [] STaSy
- [] TabDDPM
- [] TabSyn

## Contact

- Zeyu Yang
