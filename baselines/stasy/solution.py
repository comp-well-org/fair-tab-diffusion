import os
import time
import math
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import ml_collections as mlc
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

################################################################################
# data
def preprocess(data_dir):
    xn_train = pd.read_csv(os.path.join(data_dir, 'xn_train.csv'), index_col=0)
    xn_eval = pd.read_csv(os.path.join(data_dir, 'xn_eval.csv'), index_col=0)
    xn_test = pd.read_csv(os.path.join(data_dir, 'xn_test.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), index_col=0)
    y_eval = pd.read_csv(os.path.join(data_dir, 'y_eval.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), index_col=0)
    
    # only the first column of y_train, y_eval, y_test is used
    y_train = y_train.iloc[:, 0]
    y_eval = y_eval.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    
    data_train = pd.concat([xn_train, y_train], axis=1)
    data_eval = pd.concat([xn_eval, y_eval], axis=1)
    data_test = pd.concat([xn_test, y_test], axis=1)
    # print(data_train.head())
    
    with open(os.path.join(data_dir, 'desc.json')) as f:
        desc = json.load(f)
    # print(json.dumps(desc, indent=4))
    
    categories = desc['n_unq_cat_od_x_lst'] + [desc['n_unq_y']]
    d_numerical = desc['d_num_x']
    X_train_num = data_train.iloc[:, :d_numerical].values
    X_eval_num = data_eval.iloc[:, :d_numerical].values
    X_test_num = data_test.iloc[:, :d_numerical].values
    
    X_train_cat = data_train.iloc[:, d_numerical:].values
    X_eval_cat = data_eval.iloc[:, d_numerical:].values
    X_test_cat = data_test.iloc[:, d_numerical:].values
    
    # convert X_train_cat, X_eval_cat, X_test_cat to one-hot encoding
    X_train_cat = categorical_to_onehot(X_train_cat, categories)
    X_eval_cat = categorical_to_onehot(X_eval_cat, categories)
    X_test_cat = categorical_to_onehot(X_test_cat, categories)
    
    X_num_sets = (X_train_num, X_eval_num, X_test_num)
    X_cat_sets = (X_train_cat, X_eval_cat, X_test_cat)
    
    return X_num_sets, X_cat_sets, categories, d_numerical

################################################################################
# model
NONLINEARITIES = {
    'elu': nn.ELU(),
    'relu': nn.ReLU(),
    'lrelu': nn.LeakyReLU(negative_slope=0.2),
    'swish': nn.SiLU(),
    'tanh': nn.Tanh(),
    'softplus': nn.Softplus(),
}

_MODELS = {}

def register_model(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f'already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def get_sigmas(config):
    sigmas = np.exp(np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))
    return sigmas

def get_act(config):
    if config.model.activation.lower() == 'elu':
        return nn.ELU()
    elif config.model.activation.lower() == 'relu':
        return nn.ReLU()
    elif config.model.activation.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.activation.lower() == 'swish':
        return nn.SiLU()
    elif config.model.activation.lower() == 'tanh':
        return nn.Tanh()
    elif config.model.activation.lower() == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError('activation function does not exist!')

class VarianceScaling:
    def __init__(self, scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device='cpu'):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.in_axis = in_axis
        self.out_axis = out_axis
        self.dtype = dtype
        self.device = device

    def _compute_fans(self, shape):
        receptive_field_size = np.prod(shape) / shape[self.in_axis] / shape[self.out_axis]
        fan_in = shape[self.in_axis] * receptive_field_size
        fan_out = shape[self.out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(self, shape):
        fan_in, fan_out = self._compute_fans(shape)
        if self.mode == 'fan_in':
            denominator = fan_in
        elif self.mode == 'fan_out':
            denominator = fan_out
        elif self.mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f'invalid mode for variance scaling initializer: {self.mode}')
        
        variance = self.scale / denominator
        
        if self.distribution == 'normal':
            return torch.randn(*shape, dtype=self.dtype, device=self.device) * np.sqrt(variance)
        elif self.distribution == 'uniform':
            return (torch.rand(*shape, dtype=self.dtype, device=self.device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError('invalid distribution for variance scaling initializer')

def default_init(scale=1.):
    scale = 1e-10 if scale == 0 else scale
    return VarianceScaling(scale, 'fan_avg', 'uniform')

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: 
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Residual(nn.Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t, x):
        return self.bn(self._layer(x))

class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, layer_type=nn.Linear, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = layer_type(dim_in, dim_out)
        self._layer1 = layer_type(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        out = y0 + (y1 - y0) * t[:, None]
        out = self.bn(out)
        return out

class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t[:, None]
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ConcatLinearV2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))

class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(-1, 1)))

class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t.view(-1, 1))) + self._hyper_bias(t.view(-1, 1))

class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

@register_model(name='ncsnpp')
class NCSNpp(nn.Module):
    def __init__(self, config):
        super().__init__()
        base_layer = {
            'ignore': IgnoreLinear,
            'squash': SquashLinear,
            'concat': ConcatLinear,
            'concat_v2': ConcatLinearV2,
            'concatsquash': ConcatSquashLinear,
            'blend': BlendLinear,
            'concatcoord': ConcatLinear,
        }

        self.config = config
        self.act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config)))
        self.hidden_dims = config.model.hidden_dims 

        self.nf = nf = config.model.nf

        self.conditional = conditional = config.model.conditional 
        self.embedding_type = embedding_type = config.model.embedding_type.lower()

        modules = []
        if embedding_type == 'fourier':
            assert config.training.continuous, 'Fourier features are only used for continuous training.'
            modules.append(GaussianFourierProjection(embedding_size=nf, scale=config.model.fourier_scale))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            pass

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_init().init(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init().init(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        dim = config.data.image_size
        for item in list(config.model.hidden_dims):
            modules += [
                base_layer[config.model.layer_type](dim, item)
            ]
            dim += item
            modules.append(NONLINEARITIES[config.model.activation])

        modules.append(nn.Linear(dim, config.data.image_size))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond):
        modules = self.all_modules 
        m_idx = 0
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            used_sigmas = self.sigmas[time_cond.long()]
            temb = get_timestep_embedding(time_cond, self.nf)
        else:
            pass

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        temb = x
        for _ in range(len(self.hidden_dims)):
            temb1 = modules[m_idx](t=time_cond, x=temb)
            temb = torch.cat([temb1, temb], dim=1)
            m_idx += 1
            temb = modules[m_idx](temb) 
            m_idx += 1

        h = modules[m_idx](temb)
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h

def get_model(name):
    return _MODELS[name]

def create_model(config):
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)
    return score_model

################################################################################
# utils
def categorical_to_onehot(cat_matrix, categories):
    cat_matrix = cat_matrix.astype(int)
    # create a list to store the one-hot encoded values
    onehot = []
    # iterate over the columns of the categorical matrix
    for i in range((cat_matrix.shape[1])):
        # create a one-hot encoded matrix for the i-th column
        onehot_i = np.eye(categories[i])[cat_matrix[:, i]]
        # append the one-hot encoded matrix to the list
        onehot.append(onehot_i)
    # concatenate the one-hot encoded matrices along the columns
    return np.concatenate(onehot, axis=1)

def onehot_to_categorical(onehot_matrix, categories):
    # create a list to store the categorical values
    categorical = []
    # iterate over the columns of the one-hot matrix
    st = 0
    for i in range(len(categories)):
        ed = st + categories[i]
        # create a categorical matrix for the i-th column
        categorical_i = np.argmax(onehot_matrix[:, st:ed], axis=1)
        # append the categorical matrix to the list
        categorical.append(categorical_i)
        st = ed
    return np.stack(categorical, axis=1)

################################################################################
# configs
def get_default_configs():
    config = mlc.ConfigDict()

    config.seed = 42
    config.device = torch.device('cuda:1')
    config.baseline = False

    # training
    config.training = training = mlc.ConfigDict()
    config.training.batch_size = 1000
    training.epoch = 10000
    training.snapshot_freq = 300
    training.eval_freq = 100
    training.snapshot_freq_for_preemption = 100
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.eps = 1e-05
    training.loss_weighting = False
    training.spl = True
    training.lambda_ = 0.5

    #fine_tune
    training.eps_iters = 50
    training.fine_tune_epochs = 50
    training.retrain_type = 'median'
    training.hutchinson_type = 'Rademacher'
    training.tolerance = 1e-03

    # sampling
    config.sampling = sampling = mlc.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = mlc.ConfigDict()
    evaluate.num_samples = 22560

    # data
    config.data = data = mlc.ConfigDict()
    data.centered = False
    data.uniform_dequantization = False

    # model
    config.model = model = mlc.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 10.
    model.num_scales = 50
    model.alpha0 = 0.3
    model.beta0 = 0.95

    # optimization
    config.optim = optim = mlc.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-3
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    # test
    config.test = mlc.ConfigDict()

    return config

def get_config(name):
    config = get_default_configs()

    config.data.dataset = name
    config.training.batch_size = 1000
    config.eval.batch_size = 1000
    config.data.image_size = 77

    # training
    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.n_iters = 100000
    training.tolerance = 1e-03
    training.hutchinson_type = 'Rademacher'
    training.retrain_type = 'median'

    # sampling
    sampling = config.sampling
    sampling.method = 'ode'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # model
    model = config.model
    model.layer_type = 'concatsquash'
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.activation = 'elu'

    model.nf = 64
    model.hidden_dims = (1024, 2048, 1024, 1024)
    model.conditional = True
    model.embedding_type = 'fourier'
    model.fourier_scale = 16
    model.conv_size = 3

    model.sigma_min = 0.01
    model.sigma_max = 10.

    # test
    test = config.test
    test.n_iter = 1

    # optim
    optim = config.optim
    optim.lr = 2e-3

    return config

################################################################################
# training

################################################################################
# sampling

################################################################################
# main
def main():
    # global variables
    
    # TODO: configs
    dataname = 'adult'
    batch_size = 256
    
    # data
    dataset_dir = f'/rdf/db/public-tabular-datasets/{dataname}/'
    ckpt_dir = f'./ckpt/{dataname}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    X_num_sets, X_cat_sets, categories, d_numerical = preprocess(dataset_dir)
    X_train_num, X_eval_num, X_test_num = X_num_sets
    X_train_cat, X_eval_cat, X_test_cat = X_cat_sets
    X_train_num = torch.tensor(X_train_num.astype(np.float32)).float()
    X_train_cat = torch.tensor(X_train_cat.astype(np.int32)).long()
    categories = np.array(categories)
    
    train_z = torch.cat((X_train_num, X_train_cat), dim=1)
    # print(train_z.shape)
    
    # model
    config = get_config(dataname)
    score_model = create_model(config)
    num_params = sum(p.numel() for p in score_model.parameters())
    print(f'number of parameters: {num_params}')
    
    # training
    
    # sampling
    pass

if __name__ == '__main__':
    main()
