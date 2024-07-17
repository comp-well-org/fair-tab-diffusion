"""Code was adapted from https://github.com/Yura52/rtdl."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Type, Union
from cfair.seq.generator.diffusion.ddpm import timestep_embedding
from cfair.seq.generator.diffusion.configs import DenoiseFnCfg, DataCfg, GuidCfg


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def reglu(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return geglu(x)


def _make_nn_module(module_type, *args) -> nn.Module:
    if module_type == 'ReGLU':
        return ReGLU()
    elif module_type == 'GEGLU':
        return GEGLU()
    elif isinstance(module_type, str):
        return getattr(nn, module_type)(*args)
    return module_type(*args)


class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in {'ReGLU', 'GEGLU'}

        self.blocks = nn.ModuleList(
            [
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ],
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            message = 'if d_layers contains more than two elements, then all elements except for the first and the last ones must be equal.'
            assert len(set(d_layers[1:-1])) == 1, message
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPDenoiseFn(nn.Module):
    def __init__(
        self, 
        denoise_fn_cfg: DenoiseFnCfg,
        data_cfg: DataCfg,
        guid_cfg: GuidCfg,
        rtdl_params: dict,
    ) -> None:
        super().__init__()
        self.denoise_fn_cfg = copy.deepcopy(denoise_fn_cfg)
        self.data_cfg = copy.deepcopy(data_cfg)
        self.guid_cfg = copy.deepcopy(guid_cfg)
        self.guid_cfg_history = [copy.deepcopy(guid_cfg)]
        
        # embedding functions
        d_cond_emb = self.denoise_fn_cfg.d_cond_emb
        n_unq_c_lst = self.data_cfg.n_unq_c_lst 
        self.cond_emb_fn = nn.ModuleList()
        
        # label instruction
        if n_unq_c_lst[0] == 0:
            self.cond_emb_fn.append(nn.Linear(1, d_cond_emb))
        else:
            self.cond_emb_fn.append(nn.Embedding(n_unq_c_lst[0], d_cond_emb))
        
        # protected feature
        assert d_cond_emb % (len(n_unq_c_lst) - 1) == 0, f'`d_cond_emb` must be divisible by {len(n_unq_c_lst) - 1}'
        d_cond_emb_div = d_cond_emb // (len(n_unq_c_lst) - 1)
        for i in range(1, len(n_unq_c_lst)):
            if n_unq_c_lst[i] == 0:
                self.cond_emb_fn.append(nn.Linear(1, d_cond_emb_div))
            else:
                self.cond_emb_fn.append(nn.Embedding(n_unq_c_lst[i], d_cond_emb_div))
        
        # timestep embedding
        d_t_emb = self.denoise_fn_cfg.d_t_emb
        self.t_emb_fn = nn.Sequential(
            nn.Linear(d_t_emb, d_t_emb),
            nn.SiLU(),
            nn.Linear(d_t_emb, d_t_emb),
        )
        d_oh_x = self.data_cfg.d_oh_x    
        self.encoder = nn.Linear(d_oh_x, d_t_emb)
        
        # cond embedding mapping
        self.cond_proj = nn.Linear(d_cond_emb, d_t_emb)
        
        # posterior estimator
        rtdl_params = dict(rtdl_params)
        rtdl_params['d_in'] = d_t_emb
        rtdl_params['d_out'] = d_oh_x
        self.rtdl_params = rtdl_params
        self.posterior_est = MLP.make_baseline(**self.rtdl_params)
        
        # guidance weight
        cond_guid_weight = self.guid_cfg.cond_guid_weight
        overall_guid_weight = self.guid_cfg.overall_guid_weight
        
        # global 
        self.d_t_emb = d_t_emb
        self.d_cond_emb = d_cond_emb
        self.d_cond_emb_div = d_cond_emb_div
        self.cond_guid_weight = cond_guid_weight
        self.overall_guid_weight = overall_guid_weight

    def update_guid_cfg(self, guid_cfg: GuidCfg) -> None:
        self.guid_cfg_history.append(copy.deepcopy(guid_cfg))
        self.guid_cfg = copy.deepcopy(guid_cfg)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # add channel dimension
            squeeze = True
        else:
            squeeze = False
        assert len(x.shape) == 3, f'x must be 3D tensor, got {len(x.shape)}D'
        assert len(t.shape) == 1, f't must be 1D tensor, got {len(t.shape)}D'
        assert len(cond.shape) == 2, f'cond must be 2D tensor, got {len(cond.shape)}D'
        
        # timestep embedding
        t_emb = self.t_emb_fn(timestep_embedding(t, self.d_t_emb))
        
        # label embedding
        cond_emb = F.silu(self.cond_emb_fn[0](cond[:, 0].long())) * self.overall_guid_weight
        
        # protected feature embedding
        n_cond_cols = cond.shape[1]
        if n_cond_cols > 1:
            feature_emb = torch.empty((cond.shape[0], n_cond_cols - 1, self.d_cond_emb_div), device=cond.device)
            for i in range(1, cond.shape[1]):
                feature_emb[:, i - 1, :] = F.silu(self.cond_emb_fn[i](cond[:, i].long())) * self.cond_guid_weight
            feature_emb = feature_emb.reshape(feature_emb.shape[0], self.d_cond_emb)
            cond_emb += feature_emb
        # print(f'cond_emb.shape: {list(cond_emb.shape)}')
        # print(f't_emb.shape: {list(t_emb.shape)}')
        x = self.encoder(x)
        if squeeze:
            x = x.squeeze(1)  # remove channel dimension to match input
        x += t_emb + self.cond_proj(cond_emb)
        x = self.posterior_est(x)
        # print(f'x.shape: {list(x.shape)}')
        return x


def _test():
    # configs
    d_oh_x = 15
    batch_size = 4
    n_channels = 1
    d_x_emb = 256
    d_t_emb = 256
    d_cond_emb = 256
    
    # data
    x = torch.randn(batch_size, n_channels, d_oh_x)
    t = torch.randint(0, 20, (batch_size,))
    cond = torch.cat(
        [
            torch.randint(0, 2, (batch_size, 1)), 
            torch.randint(0, 3, (batch_size, 1)),
            torch.randint(0, 4, (batch_size, 1)),
        ], 
        dim=1,
    )
    
    # models
    denoise_fn_cfg = DenoiseFnCfg(
        d_x_emb=d_x_emb,
        d_t_emb=d_t_emb,
        d_cond_emb=d_cond_emb,
    )
    data_cfg = DataCfg(
        d_oh_x=d_oh_x,
        n_channels=n_channels,
        n_unq_c_lst=[2, 3, 4],
    )
    guid_cfg = GuidCfg(
        cond_guid_weight=0.5,
        cond_guid_threshold=1.0,
        cond_momentum_weight=1.0,
        cond_momentum_beta=0.2,
        warmup_steps=10,
        overall_guid_weight=1.0,
    )
    denoise_fn = MLPDenoiseFn(
        denoise_fn_cfg=denoise_fn_cfg,
        data_cfg=data_cfg,
        guid_cfg=guid_cfg,
        rtdl_params={
            'd_in': d_oh_x,
            'd_layers': [512, 512],
            'dropout': 0.,
            'd_out': d_oh_x,
        },
    )
    x = denoise_fn(x, t, cond)
    print(list(x.shape))


if __name__ == '__main__':
    _test()
