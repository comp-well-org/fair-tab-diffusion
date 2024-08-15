import os

current = os.path.dirname(os.path.realpath(__file__))

EXPS_PATH = '/rdf/experiments/fair-tab-diffusion-exps/'
DB_PATH = '/rdf/db/public-tabular-datasets/'
PLOTS_PATH = os.path.join(current, 'assess', 'plots')

# current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ARGS_DIR = os.path.join(CUR_DIR, 'args')