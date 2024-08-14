import os
import sys
import time
import math
import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import skops.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader

# getting the name of the directory where the this file is present
current = os.path.dirname(os.path.realpath(__file__))

# getting the parent directory name where the current directory is present
parent = os.path.dirname(current)

# adding the parent directory to the sys.path
sys.path.append(parent)

# importing the required files from the parent directory
from lib import load_config, copy_file, load_json
from src.evaluate.metrics import evaluate_syn_data, print_metric
from constant import DB_PATH, EXPS_PATH

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
def get_act(activation):
    if activation.lower() == 'elu':
        return nn.ELU()
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation.lower() == 'swish':
        return nn.SiLU()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'softplus':
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

class Encoder(nn.Module):
    def __init__(self, encoder_dim, tdim, activation):
        super(Encoder, self).__init__()
        self.encoding_blocks = nn.ModuleList()
        for i in range(len(encoder_dim)):
            if (i + 1) == len(encoder_dim): 
                break
            encoding_block = EncodingBlock(encoder_dim[i], encoder_dim[i+1], tdim, activation)
            self.encoding_blocks.append(encoding_block)

    def forward(self, x, t):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connection = encoding_block(x, t)
            skip_connections.append(skip_connection)
        return skip_connections, x

class EncodingBlock(nn.Module):
    def __init__(self, dim_in, dim_out, tdim, activation):
        super(EncodingBlock, self).__init__()
        self.layer1 = nn.Sequential( 
            nn.Linear(dim_in, dim_out),
            get_act(activation),
        ) 
        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, dim_out),
            get_act(activation),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            get_act(activation),
        )

    def forward(self, x, t):
        x = self.layer1(x).clone()
        x += self.temb_proj(t)
        x = self.layer2(x)
        skip_connection = x
        return x, skip_connection

class Decoder(nn.Module):
    def __init__(self, decoder_dim, tdim, activation):
        super(Decoder, self).__init__()
        self.decoding_blocks = nn.ModuleList()
        for i in range(len(decoder_dim)):
            if (i + 1) == len(decoder_dim): 
                break
            decoding_block = DecodingBlock(decoder_dim[i], decoder_dim[i + 1], tdim, activation)
            self.decoding_blocks.append(decoding_block)

    def forward(self, skip_connections, x, t):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x, t)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, dim_in, dim_out, tdim, activation):
        super(DecodingBlock, self).__init__()
        self.layer1 = nn.Sequential( 
            nn.Linear(dim_in * 2, dim_in),
            get_act(activation),
        )
        self.temb_proj = nn.Sequential(
            nn.Linear(tdim, dim_in),
            get_act(activation),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            get_act(activation),
        )

    def forward(self, skip_connection, x, t):
        x = torch.cat((skip_connection, x), dim=1)
        x = self.layer1(x).clone()
        x += self.temb_proj(t)
        x = self.layer2(x)
        return x

class TabularUnet(nn.Module):
    def __init__(self, input_size, cond_size, output_size, encoder_dim, nf, activation='relu'):
        super().__init__()

        self.embed_dim = nf
        tdim = self.embed_dim*4
        self.act = get_act(activation)

        modules = []
        modules.append(nn.Linear(self.embed_dim, tdim))
        modules[-1].weight.data = default_init().init(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)
        modules.append(nn.Linear(tdim, tdim))
        modules[-1].weight.data = default_init().init(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        cond = cond_size
        cond_out = input_size // 2
        if cond_out < 2:
            cond_out = input_size
        modules.append(nn.Linear(cond, cond_out))
        modules[-1].weight.data = default_init().init(modules[-1].weight.shape)
        nn.init.zeros_(modules[-1].bias)

        self.all_modules = nn.ModuleList(modules)
        dim_in = input_size + cond_out
        dim_out = list(encoder_dim)[0]
        
        # input layer
        self.inputs = nn.Linear(dim_in, dim_out)

        # encoder
        self.encoder = Encoder(list(encoder_dim), tdim, activation)

        dim_in = list(encoder_dim)[-1]
        dim_out = list(encoder_dim)[-1]
        
        # bottom_layer
        self.bottom_block = nn.Linear(dim_in, dim_out) 
        
        # decoder
        self.decoder = Decoder(list(reversed(encoder_dim)), tdim, activation)
        dim_in = list(encoder_dim)[0]
        dim_out = output_size
        
        # output layer
        self.outputs = nn.Linear(dim_in, dim_out)

    def forward(self, x, time_cond, cond):
        modules = self.all_modules 
        m_idx = 0

        # time embedding
        temb = get_timestep_embedding(time_cond, self.embed_dim)
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = self.act(temb)
        temb = modules[m_idx](temb)
        m_idx += 1

        # condition layer
        cond = modules[m_idx](cond)
        m_idx += 1
        x = torch.cat([x, cond], dim=1).float()
        
        # input layer
        inputs = self.inputs(x)
        skip_connections, encoding = self.encoder(inputs, temb)
        encoding = self.bottom_block(encoding)
        encoding = self.act(encoding)
        x = self.decoder(skip_connections, encoding, temb) 
        outputs = self.outputs(x)

        return outputs

def extract_continous(v, t, x_shape):
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_t, n_timesteps):
        super().__init__()
        self.model = model
        self.T = n_timesteps
        betas = torch.linspace(beta_1, beta_t, n_timesteps, dtype=torch.float64).double()
        alphas = 1. - betas
        self.register_buffer('betas', betas)
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

    def make_x_t(self, x_0_con, t, noise):
        x_t_con = (
            extract_continous(self.sqrt_alphas_bar, t, x_0_con.shape) * x_0_con +
            extract_continous(self.sqrt_one_minus_alphas_bar, t, x_0_con.shape) * noise
        )
        return x_t_con
    
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_continous(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract_continous(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_t, t, mean_type='epsilon', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = t
        self.mean_type = mean_type
        self.var_type = var_type

        betas = torch.linspace(beta_1, beta_t, t, dtype=torch.float64).double()

        alphas = 1. - betas
        self.register_buffer('betas', betas)
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:t]

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_var_clipped', torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract_continous(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract_continous(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract_continous(self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_continous(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract_continous(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, cond, trans):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract_continous(model_log_var, t, x_t.shape)

        # mean parameterization
        if self.mean_type == 'epsilon':
            eps = self.model(x_t, t, cond)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var
    
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract_discrete(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def index_to_log_onehot(x, num_classes):
    log_x = torch.log(x.float().clamp(min=1e-30))
    return log_x

class MultinomialDiffusion(torch.nn.Module):
    def __init__(
        self, num_classes, shape, denoise_fn, beta_1, beta_t, n_timesteps,
        loss_type='vb_stochastic', parametrization='x0',
    ):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        self.num_classes = num_classes 
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.shape = shape
        self.num_timesteps = n_timesteps
        self.parametrization = parametrization

        betas = torch.linspace(beta_1, beta_t, n_timesteps, dtype=torch.float64).double()
        alphas = 1. - betas
        
        alphas = np.sqrt(alphas)
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        self.num_classes_column = np.concatenate([self.num_classes[i].repeat(self.num_classes[i]) for i in range(len(self.num_classes))])
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # convert to float32 and register buffers
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(n_timesteps))
        self.register_buffer('Lt_count', torch.zeros(n_timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2))
        k = 0
        kl_list = []
        for i in self.num_classes:
            sub = kl[:, k:i + k].mean(dim=1)
            kl_list.append(sub)
            k += i
        kl = torch.stack(kl_list, 1)
        return kl
    
    def log_categorical(self, log_x_start, log_prob):
        kl = (log_x_start.exp() * log_prob)
        k = 0
        kl_list = []
        for i in self.num_classes:
            sub = kl[:, k:i + k].mean(dim=1)
            kl_list.append(sub)
            k += i
        kl = torch.stack(kl_list, 1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract_discrete(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract_discrete(self.log_1_min_alpha, t, log_x_t.shape)
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.tensor(np.log(self.num_classes_column)).to(log_1_min_alpha_t.device),
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract_discrete(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract_discrete(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.tensor(np.log(self.num_classes_column)).to(log_1_min_cumprod_alpha.device),
        )
        return log_probs

    def predict_start(self, log_x_t, t, cond_con):
        x_t = log_x_t
        out = self._denoise_fn(x_t, t, cond_con)
        assert out.size(0) == x_t.size(0)
        k = 0
        log_pred = torch.empty_like(out)
        for i in range(len(self.num_classes)):
            out_column = out[:, k:self.num_classes[i] + k]
            log_pred[:, k:self.num_classes[i] + k] = F.log_softmax(out_column, dim=1) 
            k += self.num_classes[i]
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start.to(torch.float64), log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        k = 0
        unnormed_logprobs_column_list = []
        for i in range(len(self.num_classes)):
            unnormed_logprobs_column = unnormed_logprobs[:, k:self.num_classes[i] + k]
            k += self.num_classes[i]
            for _ in range(self.num_classes[i]):
                unnormed_logprobs_column_list.append(torch.logsumexp(unnormed_logprobs_column, dim=1, keepdim=True))
        unnormed_logprobs_column_ = torch.stack(unnormed_logprobs_column_list, 1).squeeze()
        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - unnormed_logprobs_column_
        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, cond_con):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t, cond_con=cond_con)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, cond_con=cond_con)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x, t, cond_con):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, cond_con=cond_con)
        out = self.log_sample_categorical(model_log_prob).to(log_x.device)
        return out

    def log_sample_categorical(self, logits):
        full_sample = []
        k = 0
        for i in range(len(self.num_classes)):
            logits_column = logits[:, k:self.num_classes[i] + k]
            k += self.num_classes[i]
            uniform = torch.rand_like(logits_column)
            gumbel_noise = - torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + logits_column).argmax(dim=1)
            col_t = np.zeros(logits_column.shape)
            col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
            full_sample.append(col_t)
        full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0).to(log_EV_qxt_x0.device)
        return log_sample

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(torch.tensor(self.num_classes_column, device=device) * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob).mean(dim=1)
        return kl_prior

    def compute_lt(self, log_x_start, log_x_t, t, cond_con, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t,
        )
        log_model_prob, log_x_recon = self.p_pred(log_x=log_x_t, t=t, cond_con=cond_con)
        if detach_mean:
            log_model_prob = log_model_prob.detach()
        kl = self.multinomial_kl(log_true_prob, log_model_prob).mean(dim=1)
        decoder_nll = -self.log_categorical(log_x_start, log_model_prob).mean(dim=1)
        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl
        return loss, log_x_recon

################################################################################
# utils
def warmup_lr_fn(step):
    return min(step, 5000) / 5000

def infiniteloop(dataloader):
    while True:
        for _, y in enumerate(dataloader):
            yield y

def make_negative_condition(x_0_con, x_0_dis):
    device = x_0_con.device
    x_0_con = x_0_con.detach().cpu().numpy()
    x_0_dis = x_0_dis.detach().cpu().numpy()

    nsc_raw = pd.DataFrame(x_0_con)
    nsd_raw = pd.DataFrame(x_0_dis)
    nsc = np.array(nsc_raw.sample(frac=1, replace=False).reset_index(drop=True))
    nsd = np.array(nsd_raw.sample(frac=1, replace=False).reset_index(drop=True))
    ns_con = nsc
    ns_dis = nsd
    return torch.tensor(ns_con).to(device), torch.tensor(ns_dis).to(device)

def training_with(x_0_con, x_0_dis, trainer_con, trainer_dis, ns_con, ns_dis, categories, n_timesteps):
    t = torch.randint(n_timesteps, size=(x_0_con.shape[0],), device=x_0_con.device)
    pt = torch.ones_like(t).float() / n_timesteps

    # co-evolving training and predict positive samples
    noise = torch.randn_like(x_0_con)
    x_t_con = trainer_con.make_x_t(x_0_con, t, noise)
    log_x_start = torch.log(x_0_dis.float().clamp(min=1e-30))
    x_t_dis = trainer_dis.q_sample(log_x_start=log_x_start, t=t)
    eps = trainer_con.model(x_t_con, t, x_t_dis.to(x_t_con.device))
    ps_0_con = trainer_con.predict_xstart_from_eps(x_t_con, t, eps=eps)
    con_loss = F.mse_loss(eps, noise, reduction='none')
    con_loss = con_loss.mean()
    kl, ps_0_dis = trainer_dis.compute_lt(log_x_start, x_t_dis, t, x_t_con)
    ps_0_dis = torch.exp(ps_0_dis)
    kl_prior = trainer_dis.kl_prior(log_x_start)
    dis_loss = (kl / pt + kl_prior).mean()

    # negative condition -> predict negative samples
    noise_ns = torch.randn_like(ns_con)
    ns_t_con = trainer_con.make_x_t(ns_con, t, noise_ns)
    log_ns_start = torch.log(ns_dis.float().clamp(min=1e-30))
    ns_t_dis = trainer_dis.q_sample(log_x_start=log_ns_start, t=t)
    eps_ns = trainer_con.model(x_t_con, t, ns_t_dis.to(ns_t_dis.device))
    ns_0_con = trainer_con.predict_xstart_from_eps(x_t_con, t, eps=eps_ns)
    _, ns_0_dis = trainer_dis.compute_lt(log_x_start, x_t_dis, t, ns_t_con)
    ns_0_dis = torch.exp(ns_0_dis)
    
    assert not torch.isnan(x_0_con).any(), 'anchor contains nan values'
    assert not torch.isnan(ps_0_con).any(), 'positive contains nan values'
    assert not torch.isnan(ns_0_con).any(), 'negative contains nan values'

    # contrastive learning loss
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    triplet_con = triplet_loss(x_0_con, ps_0_con, ns_0_con)
    st = 0
    triplet_dis = []
    for item in categories:
        ed = st + item
        ps_dis = F.cross_entropy(ps_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')
        ns_dis = F.cross_entropy(ns_0_dis[:, st:ed], torch.argmax(x_0_dis[:, st:ed], dim=-1).long(), reduction='none')

        triplet_dis.append(max((ps_dis-ns_dis).mean() + 1, 0))
        st = ed
    triplet_dis = sum(triplet_dis)/len(triplet_dis)
    return con_loss, triplet_con, dis_loss, triplet_dis

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
# training
def train_codi_model(
    train_con_data, train_dis_data, categories,
    model_con, model_dis, trainer_con, trainer_dis,
    optim_con, optim_dis, sched_con, sched_dis, 
    lambda_con, lambda_dis, grad_clip, n_timesteps,
    total_epochs_both, batch_size_train, 
    device, ckpt_dir,
):
    total_steps_both = total_epochs_both * int(train_con_data.shape[0] / batch_size_train + 1)
    train_iter_con = DataLoader(train_con_data, batch_size=batch_size_train)
    train_iter_dis = DataLoader(train_dis_data, batch_size=batch_size_train)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    best_loss = float('inf')
    for i in range(total_steps_both):
        model_con.train()
        model_dis.train()

        x_0_con = next(datalooper_train_con).to(device).float()
        x_0_dis = next(datalooper_train_dis).to(device)

        ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
        con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
            x_0_con, x_0_dis, trainer_con, trainer_dis, ns_con, ns_dis, categories, n_timesteps,
        )
        
        loss_con = con_loss + lambda_con * con_loss_ns
        loss_dis = dis_loss + lambda_dis * dis_loss_ns

        optim_con.zero_grad()
        loss_con.backward()
        torch.nn.utils.clip_grad_norm_(model_con.parameters(), grad_clip)
        optim_con.step()
        sched_con.step()

        optim_dis.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_value_(trainer_dis.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), grad_clip)
        optim_dis.step()
        sched_dis.step()
        
        total_loss = loss_con.item() + loss_dis.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model_con.state_dict(), f'{ckpt_dir}/model_con.pt')
            torch.save(model_dis.state_dict(), f'{ckpt_dir}/model_dis.pt')
        
        # if continous loss or discrete loss is nan, break the training 
        if torch.isnan(loss_con).any() or torch.isnan(loss_dis).any():
            print(f'training -> [{i+1}/{total_steps_both}], mloss: {loss_con.item():.4f}, dloss: {loss_dis.item():.4f}, tloss: {total_loss:.4f} -- best: {best_loss:.4f}')
            print('loss is nan, break the training')
            break
        
        if i == total_steps_both - 1:
            print(f'training -> [{i+1}/{total_steps_both}], mloss: {loss_con.item():.4f}, dloss: {loss_dis.item():.4f}, tloss: {total_loss:.4f} -- best: {best_loss:.4f}')
        else:
            print(f'training -> [{i+1}/{total_steps_both}], mloss: {loss_con.item():.4f}, dloss: {loss_dis.item():.4f}, tloss: {total_loss:.4f} -- best: {best_loss:.4f}', end='\r')

################################################################################
# sampling
def log_sample_categorical(logits, num_classes):
    full_sample = []
    k = 0
    for i in range(len(num_classes)):
        logits_column = logits[:, k:num_classes[i] + k]
        k += num_classes[i]
        uniform = torch.rand_like(logits_column)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits_column).argmax(dim=1)
        col_t = np.zeros(logits_column.shape)
        col_t[np.arange(logits_column.shape[0]), sample.detach().cpu()] = 1
        full_sample.append(col_t)
    full_sample = torch.tensor(np.concatenate(full_sample, axis=1))
    log_sample = torch.log(full_sample.float().clamp(min=1e-30))
    return log_sample

def sampling_with(x_t_con_end, log_x_t_dis_end, net_sampler, trainer_dis, trans, n_timesteps):
    x_t_con = x_t_con_end
    x_t_dis = log_x_t_dis_end

    for time_step in reversed(range(n_timesteps)):
        t = x_t_con.new_ones((x_t_con.shape[0],), dtype=torch.long) * time_step
        mean, log_var = net_sampler.p_mean_variance(x_t=x_t_con, t=t, cond=x_t_dis.to(x_t_con.device), trans=trans)
        if time_step > 0:
            noise = torch.randn_like(x_t_con)
        elif time_step == 0:
            noise = 0
        x_t_minus_1_con = mean + torch.exp(0.5 * log_var) * noise
        x_t_minus_1_con = torch.clip(x_t_minus_1_con, -1., 1.)
        x_t_minus_1_dis = trainer_dis.p_sample(x_t_dis, t, x_t_con)
        x_t_con = x_t_minus_1_con
        x_t_dis = x_t_minus_1_dis

    return x_t_con, x_t_dis

def sampling_synthetic_data(
    model_con, model_dis, trainer_dis, ckpt_dir, 
    train_con_data, train_dis_data,
    num_class, net_sampler, n_timesteps,
    device,
):
    model_con.load_state_dict(torch.load(f'{ckpt_dir}/model_con.pt'))
    model_dis.load_state_dict(torch.load(f'{ckpt_dir}/model_dis.pt'))

    model_con.eval()
    model_dis.eval()
    
    with torch.no_grad():
        x_T_con = torch.randn(train_con_data.shape[0], train_con_data.shape[1]).to(device)
        log_x_T_dis = log_sample_categorical(torch.zeros(train_dis_data.shape, device=device), num_class).to(device)
        x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, num_class, n_timesteps)
    
    sample_con = x_con.detach().cpu().numpy()
    sample_dis = x_dis.detach().cpu().numpy()
    sample_dis = onehot_to_categorical(sample_dis, num_class)
    sample = np.concatenate([sample_con, sample_dis], axis=1)
    return sample

################################################################################
# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--exp_name', type=str, default='check')
    parser.add_argument('--train', action='store_true', help='training', default=True)
    parser.add_argument('--sample', action='store_true', help='sampling', default=True)
    parser.add_argument('--eval', action='store_true', help='evaluation', default=True)
    
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError('config file is required')
    
    # configs
    exp_config = config['exp']
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    sample_config = config['sample']
    eval_config = config['eval']
    
    device = torch.device(exp_config['device'])
    seed = exp_config['seed']
    torch.manual_seed(seed)
    
    batch_size = data_config['batch_size']
    
    encoder_dim_con = model_config['encoder_dim_con']
    nf_con = model_config['nf_con']
    lr_con = model_config['lr_con']
    beta_1 = model_config['beta_1']
    beta_t = model_config['beta_t']
    n_timesteps = model_config['n_timesteps']
    encoder_dim_dis = model_config['encoder_dim_dis']
    nf_dis = model_config['nf_dis']
    lr_dis = model_config['lr_dis']
    
    lambda_con = train_config['lambda_con']
    lambda_dis = train_config['lambda_dis']
    grad_clip = train_config['grad_clip']
    total_epochs_both = train_config['total_epochs_both']
    
    mean_type = sample_config['mean_type']
    var_type = sample_config['var_type']
    n_seeds = sample_config['n_seeds']
    
    # message
    print(f'config file: {args.config}')
    print('-' * 80)
    
    # experimental directory
    exp_dir = os.path.join(
        exp_config['home'], 
        data_config['name'],
        exp_config['method'],
        args.exp_name,
    )
    copy_file(
        os.path.join(exp_dir), 
        args.config,
    )
    
    # data
    dataset_dir = os.path.join(data_config['path'], data_config['name'])
    data_desc = load_json(os.path.join(dataset_dir, 'desc.json'))
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    X_num_sets, X_cat_sets, categories, d_numerical = preprocess(dataset_dir)
    X_train_num, X_eval_num, X_test_num = X_num_sets
    X_train_cat, X_eval_cat, X_test_cat = X_cat_sets
    X_train_num = torch.tensor(X_train_num.astype(np.float32)).float()
    X_train_cat = torch.tensor(X_train_cat.astype(np.int32)).long()
    categories = np.array(categories)
    norm_fn = sio.load(os.path.join(dataset_dir, 'fn.skops'))
    feature_cols = data_desc['col_names']
    label_cols = [data_desc['label_col_name']]
    
    # model
    input_size = X_train_num.shape[1] 
    cond_size = X_train_cat.shape[1]
    output_size = X_train_num.shape[1]
    model_con = TabularUnet(input_size, cond_size, output_size, encoder_dim_con, nf_con)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr_fn)
    trainer_con = GaussianDiffusionTrainer(
        model_con, beta_1, beta_t, n_timesteps,
    ).to(device)
    
    input_size = X_train_cat.shape[1] 
    cond_size = X_train_num.shape[1]
    output_size = X_train_cat.shape[1]
    model_dis = TabularUnet(input_size, cond_size, output_size, encoder_dim_dis, nf_dis)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr_fn)
    trainer_dis = MultinomialDiffusion(
        categories, X_train_cat.shape, model_dis, beta_1, beta_t, n_timesteps,
    ).to(device)

    num_params_con = sum(p.numel() for p in model_con.parameters())
    num_params_dis = sum(p.numel() for p in model_dis.parameters())
    num_params = num_params_con + num_params_dis
    with open(os.path.join(exp_dir, 'params.txt'), 'w') as f:
        f.write(f'number of parameters: {num_params}')

    if args.train:
        # train
        start_time = time.time()
        train_codi_model(
            X_train_num, X_train_cat, categories,
            model_con, model_dis, trainer_con, trainer_dis,
            optim_con, optim_dis, sched_con, sched_dis, 
            lambda_con, lambda_dis, grad_clip, n_timesteps,
            total_epochs_both, batch_size, 
            device, ckpt_dir,
        )
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'w') as f:
            time_msg = f'training time: {end_time - start_time:.2f} seconds with {total_epochs_both} epochs'
            f.write(time_msg)
        print()
            
    if args.sample:
        # sampling
        net_sampler = GaussianDiffusionSampler(
            model_con, beta_1, beta_t, n_timesteps, mean_type, var_type,
        ).to(device)
        
        start_time = time.time()
        for i in range(n_seeds):
            random_seed = seed + i
            torch.manual_seed(random_seed)
            sample = sampling_synthetic_data(
                model_con, model_dis, trainer_dis, ckpt_dir, 
                X_train_num, X_train_cat,
                categories, net_sampler, n_timesteps,
                device,
            )
            # xn + xd + y -> [x + xd] + y
            xn_num = sample[:, :d_numerical]
            x_num = norm_fn.inverse_transform(sample[:, :d_numerical])
            x_cat = sample[:, d_numerical: -1]
            xn_syn = np.concatenate([xn_num, x_cat], axis=1)
            x_syn = np.concatenate([x_num, x_cat], axis=1)
            y_syn = sample[:, -1]
            
            # to dataframe
            xn_syn = pd.DataFrame(xn_syn, columns=feature_cols)
            x_syn = pd.DataFrame(x_syn, columns=feature_cols)
            y_syn = pd.DataFrame(y_syn, columns=label_cols)

            synth_dir = os.path.join(exp_dir, f'synthesis/{random_seed}')
            if not os.path.exists(synth_dir):
                os.makedirs(synth_dir)
            
            x_syn.to_csv(os.path.join(synth_dir, 'x_syn.csv'))
            xn_syn.to_csv(os.path.join(synth_dir, 'xn_syn.csv'))
            y_syn.to_csv(os.path.join(synth_dir, 'y_syn.csv'))
            print(f'seed: {random_seed}, xn_syn: {xn_syn.shape}, y_syn: {y_syn.shape}')
        
        end_time = time.time()
        with open(os.path.join(exp_dir, 'time.txt'), 'a') as f:
            time_msg = f'\nsampling time: {end_time - start_time:.2f} seconds with {n_seeds} seeds'
            f.write(time_msg)
        print

    if args.eval:
        # evaluate classifiers trained on synthetic data
        synth_dir_list = []
        for i in range(n_seeds):
            synth_dir = os.path.join(exp_dir, f'synthesis/{seed + i}')
            if os.path.exists(synth_dir):
                synth_dir_list.append(synth_dir)

        sst_col_names = data_desc['sst_col_names']
        metric = evaluate_syn_data(
            data_dir=os.path.join(data_config['path'], data_config['name']),
            exp_dir=exp_dir,
            synth_dir_list=synth_dir_list,
            sk_clf_lst=eval_config['sk_clf_choice'],
            sens_cols=sst_col_names,
        )

        with open(os.path.join(exp_dir, 'metric.json'), 'w') as f:
            json.dump(metric, f, indent=4)
            
        # print metric
        print_metric(metric)

if __name__ == '__main__':
    main()
