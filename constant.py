import os

EXPS_PATH = '/rdf/experiments/fair-tab-diffusion-exps/'
DB_PATH = '/rdf/db/public-tabular-datasets/'
ADULT_PATH = os.path.join(DB_PATH, 'adult')

# current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ARGS_DIR = os.path.join(CUR_DIR, 'args')