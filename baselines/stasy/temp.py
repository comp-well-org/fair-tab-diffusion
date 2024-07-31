import torch
import numpy as np
import torch.nn as nn
import math

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


def variance_scaling(
    scale, mode, distribution,
    in_axis=1, out_axis=0,
    dtype=torch.float32,
    device='cpu',
):
    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == 'fan_in':
            denominator = fan_in
        elif mode == 'fan_out':
            denominator = fan_out
        elif mode == 'fan_avg':
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError('invalid mode for variance scaling initializer: {}'.format(mode))
        variance = scale / denominator
        if distribution == 'normal':
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == 'uniform':
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError('invalid distribution for variance scaling initializer')
        return init

def default_init(scale=1.):
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')

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
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_init()(modules[-1].weight.shape)
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