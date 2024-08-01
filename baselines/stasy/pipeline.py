import os
import abc
import time
import math
import json
import torch
import logging
import warnings
import functools
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ml_collections as mlc
from scipy import integrate
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
                base_layer[config.model.layer_type](dim, item),
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
def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))

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

def onehot_argmax_categorical(onehot_matrix, categories):
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

class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates, shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

def get_model_fn(model, train=False):
    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)
    return model_fn

def get_score_fn(sde, model, train=False, continuous=False):
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
        def score_fn(x, t):
            if continuous or isinstance(sde, SubVPSDE):
                labels = t * (sde.N - 1)
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                labels = t * (sde.N - 1)
                score = model_fn(x, labels) 
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
                score = -score / std[:, None]
            return score
    elif isinstance(sde, VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1] 
            else:
                labels = sde.T - t
                labels *= sde.N - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score
    else:
        raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')
    return score_fn

def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step'],
        'epoch': state['epoch'],
    }
    torch.save(saved_state, ckpt_dir)

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    state['epoch'] = loaded_state['epoch']
    return state

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

    # finetune
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
# loss
def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(f'optimizer {config.optim.optimizer} not supported yet!')
    return optimizer

def optimization_manager(config):
    def optimize_fn(
        optimizer, params, step, lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum((step+1) / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()
    return optimize_fn

def get_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    
    def loss_fn(model, batch):
        score_fn = get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z
        score = score_fn(perturbed_data, t)
        if not likelihood_weighting:
            losses = torch.square(score * std[:, None] + z) 
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        return losses

    return loss_fn

def get_smld_loss_fn(vesde, train, reduce_mean=False):
    assert isinstance(vesde, VESDE), 'SMLD training only works for VESDEs'

    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train)

        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels)
        target = -noise / (sigmas ** 2)[:, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn

def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    assert isinstance(vpsde, VPSDE), 'DDPM training only works for VPSDEs'
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        model_fn = get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + sqrt_1m_alphas_cumprod[labels, None] * noise
        score = model_fn(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn

def min_max_scaling(factor, scale=(0, 1)):
    std = (factor - factor.min()) / (factor.max() - factor.min())
    new_min = torch.tensor(scale[0]) 
    new_max = torch.tensor(scale[1])
    return std * (new_max - new_min) + new_min

def compute_v(ll, alpha, beta):
    v = -torch.ones(ll.shape).to(ll.device)
    v[torch.gt(ll, beta)] = torch.tensor(0., device=v.device) 
    v[torch.le(ll, alpha)] = torch.tensor(1., device=v.device)
    if ll[torch.eq(v, -1)].shape[0] != 0 and ll[torch.eq(v, -1)].shape[0] != 1:
        v[torch.eq(v, -1)] = min_max_scaling(ll[torch.eq(v, -1)], scale=(1, 0)).to(v.device)
    else:
        v[torch.eq(v, -1)] = torch.tensor(0.5, device=v.device)
    return v  

def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False, workdir=False, spl=True, writer=None, alpha0=None, beta0=None):
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, continuous=True, likelihood_weighting=likelihood_weighting)
    else:
        assert not likelihood_weighting, 'likelihood weighting is not supported for original SMLD/DDPM training'
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f'Discrete training for {sde.__class__.__name__} is not recommended.')

    def step_fn(state, batch):
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            losses = loss_fn(model, batch)
            if spl:
                nll = losses
                q_alpha = torch.tensor(alpha0 + torch.log(torch.tensor(1 + 0.0001718 * state['step'] * (1 - alpha0), dtype=torch.float32))).clamp_(max=1).to(nll.device)
                q_beta = torch.tensor(beta0 + torch.log(torch.tensor(1 + 0.0001718 * state['step'] * (1 - beta0), dtype=torch.float32))).clamp_(max=1).to(nll.device)
                logging.info(f'q_alpha: {q_alpha}, q_beta: {q_beta}')

                alpha = torch.quantile(nll, q_alpha) 
                beta = torch.quantile(nll, q_beta)
                assert alpha <= beta
                v = compute_v(nll, alpha, beta)
                loss = torch.mean(v*losses)
                
                logging.info(f'alpha: {alpha}, beta: {beta}')
                logging.info(f'1 samples: {torch.sum(v == 1)} / {len(v)}')
                logging.info(f'weighted samples: { torch.sum((v != 1) * (v != 0)  )} / {len(v)}')
                logging.info(f'0 samples: {torch.sum(v == 0)} / {len(v)}')
            else:
                loss = torch.mean(losses)

            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                losses, score = loss_fn(model, batch)
                ema.restore(model.parameters())
                loss = torch.mean(losses)
        return loss
    return step_fn

################################################################################
# sdes
class SDE(abc.ABC):
    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
            
        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x

        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        logZ = -0.5 * np.log(2 * np.pi)
        return logZ - z ** 2 / 2

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None] * x - x

        G = sqrt_beta
        return f, G

class SubVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x

        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)[:, None] * x

        std = 1 - torch.exp(2. * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        logZ = -0.5 * np.log(2 * np.pi)
        return logZ - z ** 2 / 2

class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=1) / (2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]

        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), self.discrete_sigmas.to(t.device)[timestep - 1].to(t.device))
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G

################################################################################
# sampling
_CORRECTORS = {}
_PREDICTORS = {}

def register_predictor(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def get_predictor(name):
    return _PREDICTORS[name]

def get_corrector(name):
    return _CORRECTORS[name]

def get_sampling_fn(config, sde, shape, eps):
    sampler_name = config.sampling.method
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    else:
        raise ValueError(f'sampler name {sampler_name} unknown.')
    return sampling_fn

class Predictor(abc.ABC):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

class Corrector(abc.ABC):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
        return x, x_mean

@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None] * z
        return x, x_mean

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')
        assert not probability_flow, 'probability flow not supported by ancestral sampling'

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep].to(t.device)
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)

        x_mean = (x + beta[:, None] * score) / torch.sqrt(1. - beta)[:, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None] * noise

        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(x, t)

@register_predictor(name='none')
class NonePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE) and not isinstance(sde, SubVPSDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size[:, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise

        return x, x_mean

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE) and not isinstance(sde, SubVPSDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None]

        return x, x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)

def get_pc_sampler(
    sde, shape, predictor, corrector, snr,
    n_steps=1, probability_flow=False, continuous=False,
    denoise=True, eps=1e-3, device='cuda',
):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(model, sampling_shape=None):
        if sampling_shape:
            shape = sampling_shape

        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

            return x_mean if denoise else x, sde.N * (n_steps + 1)

    return pc_sampler

def get_ode_sampler(
    sde, shape,
    denoise=False, rtol=1e-5, atol=1e-5,
    method='RK45', eps=1e-3, device='cuda',
):

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None, sampling_shape=None):
        if sampling_shape:
            shape = sampling_shape
        
        with torch.no_grad():
            if z is None:
                x = sde.prior_sampling(shape).to(device)
                start = eps
                end = sde.T
            else:
                shape = z.shape
                x = z
                start = sde.T
                end = sde.T + 1e-08

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            solution = integrate.solve_ivp(ode_func, (end, start), to_flattened_numpy(x), rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            if denoise:
                x = denoise_update_fn(model, x)
            return x, nfe

    return ode_sampler

################################################################################
# main
def main():
    # global variables
    
    # TODO: configs
    dataname = 'adult'
    n_epochs = 2
    n_samples = 1000
    
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
    
    # update config
    config = get_config(dataname)
    config.data.image_size = train_z.shape[1]
    config.training.epoch = n_epochs
    
    # model
    score_model = create_model(config)
    num_params = sum(p.numel() for p in score_model.parameters())
    print(f'number of parameters: {num_params}')
    
    # training
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    
    optimizer = get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)

    initial_step = int(state['epoch'])
    train_data = train_z
    train_iter = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True, num_workers=4)
    
    # set up sdes
    if config.training.sde.lower() == 'vpsde':
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'subvpsde':
        sde = SubVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    elif config.training.sde.lower() == 'vesde':
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f'SDE {config.training.sde} unknown')
    
    optimize_fn = optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    
    train_step_fn = get_step_fn(
        sde, train=True, optimize_fn=optimize_fn,
        reduce_mean=reduce_mean, continuous=continuous,
        likelihood_weighting=likelihood_weighting, workdir=ckpt_dir, spl=config.training.spl, 
        alpha0=config.model.alpha0, beta0=config.model.beta0,
    )

    start_time = time.time()
    best_loss = np.inf
    for epoch in range(initial_step, config.training.epoch):
        state['epoch'] += 1

        batch_loss = 0
        batch_num = 0
        epoch_loss = 0
        for idx, batch in enumerate(train_iter): 
            batch = batch.to(config.device).float()
            num_sample = batch.shape[0]
            batch_num += num_sample
            loss = train_step_fn(state, batch)
            batch_loss += loss.item() * num_sample
            epoch_loss = batch_loss / batch_num

            if epoch == config.training.epoch - 1 and idx == len(train_iter) - 1:
                print(f'training -> epoch: {epoch + 1}/{config.training.epoch}, loss: {epoch_loss:.4f} -- best: {best_loss:.4f}')
            else:
                print(f'training -> epoch: {epoch + 1}/{config.training.epoch}, loss: {epoch_loss:.4f} -- best: {best_loss:.4f}', end='\r')
        
        batch_loss /= batch_num
        if batch_loss < best_loss:
            best_loss = batch_loss
            save_checkpoint(os.path.join(ckpt_dir, 'model.pth'), state)
            
    end_time = time.time()
    print(f'training time: {(end_time - start_time):.2f}s')
    
    # sampling
    state = restore_checkpoint(os.path.join(ckpt_dir, 'model.pth'), state, config.device)
    sampling_fn = get_sampling_fn(config, sde, shape=train_z.shape, eps=1e-3)
    samples, n = sampling_fn(score_model, sampling_shape=(n_samples, train_z.shape[1]))
    syn_data_num = samples[:, :d_numerical].detach().cpu().numpy()
    syn_data_cat = samples[:, d_numerical:].detach().cpu().numpy()
    syn_data_cat = onehot_argmax_categorical(syn_data_cat, categories)
    print(syn_data_num.shape, syn_data_cat.shape)
    syn_data = np.concatenate((syn_data_num, syn_data_cat), axis=1)
    syn_data = pd.DataFrame(syn_data)
    print(syn_data.head(3))

if __name__ == '__main__':
    main()
