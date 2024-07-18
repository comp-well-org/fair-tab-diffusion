import copy
import math
import torch
import torch.nn as nn
from .tools import timestep_embedding
from .configs import DenoiseFnCfg, DataCfg, GuidCfg
from .unet import Unet

class PosteriorEstimator(nn.Module):
    def __init__(self, pstr_est: Unet) -> None:
        super().__init__()
        self.pstr_est = pstr_est
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, sqrt(d_x_emb), sqrt(d_x_emb)]`.
            t: timestep tensor of shape `[batch_size, d_t_emb]`.
            cond: condition tensor of shape `[batch_size, n_cond, d_cond_emb]`.
        
        Returns:
            x: output tensor of shape `[batch_size, n_channels, sqrt(d_x_emb), sqrt(d_x_emb)]`.
        """
        return self.pstr_est(x, t, cond)

class DenoiseFn(nn.Module):
    def __init__(
        self, 
        denoise_fn_cfg: DenoiseFnCfg, 
        data_cfg: DataCfg,
        guid_cfg: GuidCfg,
        encoder: callable = None, 
        decoder: callable = None, 
        posterior_est: PosteriorEstimator = None,
    ) -> None:
        super().__init__()
        # config
        self.denoise_fn_cfg = copy.deepcopy(denoise_fn_cfg)
        self.data_cfg = copy.deepcopy(data_cfg)
        self.guid_cfg = copy.deepcopy(guid_cfg)
        self.guid_cfg_history = [copy.deepcopy(guid_cfg)]
        
        # embedding functions
        d_cond_emb = self.denoise_fn_cfg.d_cond_emb
        n_unq_c_lst = self.data_cfg.n_unq_c_lst
        self.uncond_emb_fn = nn.Embedding(1, d_cond_emb)        
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
        
        d_t_emb = self.denoise_fn_cfg.d_t_emb
        self.t_emb_fn = nn.Sequential(
            nn.Linear(d_t_emb, d_t_emb),
            nn.SiLU(),
            nn.Linear(d_t_emb, d_t_emb),
        )
        
        # encoder, decoder, posterior estimator
        d_oh_x = self.data_cfg.d_oh_x
        n_channels = self.data_cfg.n_channels
        d_x_emb = self.denoise_fn_cfg.d_x_emb
        if encoder is None:
            self.encoder = nn.Sequential(
                nn.Linear(d_oh_x, d_x_emb),
                nn.SiLU(),
                nn.Linear(d_x_emb, d_x_emb),
            )
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = nn.Sequential(
                nn.Linear(d_x_emb, d_oh_x),
                nn.SiLU(),
                nn.Linear(d_oh_x, d_oh_x),
            )
        else:
            self.decoder = decoder
        self.posterior_est = posterior_est
        
        # global 
        self.d_x_emb = d_x_emb
        self.d_t_emb = d_t_emb
        self.d_cond_emb = d_cond_emb
        self.d_cond_emb_div = d_cond_emb_div
        self.n_channels = n_channels
    
    def update_guid_cfg(self, guid_cfg: GuidCfg) -> None:
        self.guid_cfg_history.append(copy.deepcopy(guid_cfg))
        self.guid_cfg = copy.deepcopy(guid_cfg)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, d_oh_x]`.
            t: timestep tensor of shape `[batch_size,]`.
            cond: condition tensor of shape `[batch_size, n_cond]`.
        
        Returns:
            x: output tensor of shape `[batch_size, n_channels, d_oh_x]`.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # add channel dimension
            squeeze = True
            assert self.n_channels == 1, f'x has shape {x.shape}, but n_channels is {self.n_channels}'
        else:
            squeeze = False
        assert len(x.shape) == 3, f'x must be 3D tensor, got {len(x.shape)}D'
        assert len(t.shape) == 1, f't must be 1D tensor, got {len(t.shape)}D'
        assert len(cond.shape) == 2, f'cond must be 2D tensor, got {len(cond.shape)}D'
        # encode x of shape `[batch_size, n_channels, d_oh_x]`
        x = self.encoder(x)
        # print(f'x.shape: {x.shape} after `encoder`')
        n_cond_cols = cond.shape[1]
        if n_cond_cols > 1:
            coef = 2
        elif n_cond_cols == 1:
            coef = 1
        x = torch.cat([x] * (1 + coef), dim=2)  # repeat in feature dimension
        # print(f'x.shape: {x.shape} after `repeat`')
        
        # reshape `x` to `[batch_size, n_channels, sqrt(d_x_emb), sqrt(d_x_emb)]`
        sqrt_d_x_emb = int(math.sqrt(self.d_x_emb))
        x = x.reshape(x.shape[0], self.n_channels, sqrt_d_x_emb, sqrt_d_x_emb * (1 + coef))
        # print(f'x.shape: {x.shape} after reshape')
        
        # t_emb of shape `[batch_size, d_t_emb]`
        t_emb = self.t_emb_fn(timestep_embedding(t, self.d_t_emb))
        # print(f't_emb.shape: {t_emb.shape} after `t_emb_fn`')
        
        # cond_emb of shape `[batch_size, n_cond, d_cond_emb]`
        # cond_emb_lst = []
        cond_emb = torch.empty((cond.shape[0], 1 + coef, self.d_cond_emb), device=cond.device)
        
        # unconditional part
        zeros = torch.zeros(cond.shape[0]).long().to(cond.device)
        cond_emb[:, 0, :] = self.uncond_emb_fn(zeros)

        # label instruction
        cond_emb[:, 1, :] = self.cond_emb_fn[0](cond[:, 0].long())
        
        # protected feature embedding fusion
        if n_cond_cols > 1:
            feature_emb = torch.empty((cond.shape[0], n_cond_cols - 1, self.d_cond_emb_div), device=cond.device)
            for i in range(1, cond.shape[1]):
                feature_emb[:, i - 1, :] = self.cond_emb_fn[i](cond[:, i].long())
                # concat channel dimension to feature dimension
            feature_emb = feature_emb.reshape(feature_emb.shape[0], 1, self.d_cond_emb)
            cond_emb[:, 2, :] = feature_emb.squeeze(1)
        
        # concat channel dimension to feature dimension
        cond_emb = cond_emb.reshape(cond_emb.shape[0], 1, self.d_cond_emb * (1 + coef))
        
        x = self.posterior_est(x, t_emb, cond_emb)
        # print(f'x.shape: {x.shape} after `posterior_est`')
        
        # reshape `x` to `[batch_size, n_channels, d_x_emb]`
        x = x.reshape(x.shape[0], self.n_channels, self.d_x_emb * (1 + coef))
        # print(f'x.shape: {x.shape} after reshape')
        
        # classifier free guidance
        x = x.chunk((1 + coef), dim=2)
        
        # embedding decomposition
        if len(x) == 2:
            x_uncond_emb, x_label_emb = x
            x_cond_emb = None
        elif len(x) == 3:
            x_uncond_emb, x_label_emb, x_cond_emb = x
        
        # label guidance
        # $\epsilon_\theta(\mathbf{z}_t, \mathbf{c}_p)-\epsilon_\theta(\mathbf{z}_t)$
        guid_emb = x_label_emb - x_uncond_emb
        
        # condition guidance
        if x_cond_emb is not None:
            scale = torch.clamp(
                torch.abs(x_label_emb - x_cond_emb) * self.guid_cfg.cond_guid_weight,
                max=1.0,
            )
            
            cond_guid_scale = torch.where(
                (x_label_emb - x_cond_emb) > self.guid_cfg.cond_guid_threshold,
                torch.zeros_like(scale),
                scale,
            )
            
            cond_guid_emb = torch.mul(
                x_cond_emb - x_uncond_emb, cond_guid_scale,
            )
            # print(f'cond_guid_emb.shape: {list(cond_guid_emb.shape)}')
            
            cond_guid_emb_momentum = torch.zeros_like(cond_guid_emb)
            cond_guid_emb += cond_guid_emb_momentum * self.guid_cfg.cond_momentum_weight
            
            # update momentum
            beta = self.guid_cfg.cond_momentum_beta
            cond_guid_emb_momentum += beta * cond_guid_emb_momentum + (1 - beta) * cond_guid_emb
            # print(f'cond_guid_emb_momentum.shape: {list(cond_guid_emb_momentum.shape)}')
            
            # warm up
            warm_mask = torch.where(t >= self.guid_cfg.warmup_steps, 1, 0)  
            zero_indices = torch.where(warm_mask == 0)[0]
            cond_guid_emb[zero_indices] = 0.
            guid_emb += cond_guid_emb
        
        x = x_uncond_emb + guid_emb * self.guid_cfg.overall_guid_weight
        # print(f'x.shape: {x.shape} after `guidance`')
        x = self.decoder(x)
        # print(f'x.shape: {x.shape} after `decoder`')
        if squeeze:
            x = x.squeeze(1)  # remove channel dimension to match input
        return x

def _test():
    # configs
    d_oh_x = 15
    batch_size = 4
    n_channels = 1
    d_x_emb = 256
    d_t_emb = 256
    d_cond_emb = 256
    n_base_channels = 32
    n_groups = 1
    
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
    print(f'x.shape: {x.shape}, t.shape: {t.shape}, cond.shape: {cond.shape}')
    
    # fair
    coef = 3 if cond.shape[1] > 1 else 2
    
    # models
    unet = Unet(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        n_base_channels=n_base_channels,
        n_channels_factors=[2, 2, 2],
        n_res_blocks=1,
        attention_levels=[0],
        d_t_emb=d_t_emb,
        d_cond_emb=d_cond_emb * coef,
        n_groups=n_groups,
    )
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
    denoise_fn = DenoiseFn(
        denoise_fn_cfg=denoise_fn_cfg,
        data_cfg=data_cfg,
        guid_cfg=guid_cfg,
        posterior_est=PosteriorEstimator(unet),
    )
    x = denoise_fn(x, t, cond)
    print(f'x.shape: {x.shape}')

if __name__ == '__main__':
    _test()
