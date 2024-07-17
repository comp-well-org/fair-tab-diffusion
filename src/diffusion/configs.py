"""Configs."""

import math
from typing import List

class DenoiseFnCfg:    
    def __init__(self, d_x_emb: int, d_t_emb: int, d_cond_emb: int) -> None:
        assert d_x_emb > 1, f'd_x_emb must be greater than 1, got {d_x_emb}'
        assert math.log(d_x_emb, 4).is_integer(), f'd_x_emb must be positive power of 4, got {d_x_emb}'
        self._config = {}
        self._config['d_x_emb'] = d_x_emb
        self._config['d_t_emb'] = d_t_emb
        self._config['d_cond_emb'] = d_cond_emb
    
    @property
    def d_x_emb(self) -> int:
        return self._config['d_x_emb']
    
    @d_x_emb.setter
    def d_x_emb(self, v: int) -> None:
        self._config['d_x_emb'] = v
        
    @property
    def d_t_emb(self) -> int:
        return self._config['d_t_emb']
    
    @d_t_emb.setter
    def d_t_emb(self, v: int) -> None:
        self._config['d_t_emb'] = v
        
    @property
    def d_cond_emb(self) -> int:
        return self._config['d_cond_emb']
    
    @d_cond_emb.setter
    def d_cond_emb(self, v) -> None:
        self._config['d_cond_emb'] = v


class DataCfg:
    def __init__(self, n_channels: int, d_oh_x: int, n_unq_c_lst: List[int]) -> None:
        self._config = {}
        self._config['n_channels'] = n_channels
        self._config['d_oh_x'] = d_oh_x
        self._config['n_unq_c_lst'] = n_unq_c_lst
    
    @property
    def n_channels(self) -> int:
        return self._config['n_channels']
    
    @n_channels.setter
    def n_channels(self, v: int) -> None:
        self._config['n_channels'] = v
        
    @property
    def d_oh_x(self) -> int:
        return self._config['d_oh_x']
    
    @d_oh_x.setter
    def d_oh_x(self, v: int) -> None:
        self._config['d_oh_x'] = v
        
    @property
    def n_unq_c_lst(self) -> List[int]:
        return self._config['n_unq_c_lst']
    
    @n_unq_c_lst.setter
    def n_unq_c_lst(self, v: List[int]) -> None:
        self._config['n_unq_c_lst'] = v


class GuidCfg:
    def __init__(
        self, 
        cond_guid_weight: float, 
        cond_guid_threshold: float,
        cond_momentum_weight: float,
        cond_momentum_beta: float,
        warmup_steps: int,
        overall_guid_weight: float,
    ) -> None:
        self._config = {}
        self._config['cond_guid_weight'] = cond_guid_weight
        self._config['cond_guid_threshold'] = cond_guid_threshold
        self._config['cond_momentum_weight'] = cond_momentum_weight
        self._config['cond_momentum_beta'] = cond_momentum_beta
        self._config['warmup_steps'] = warmup_steps
        self._config['overall_guid_weight'] = overall_guid_weight
    
    @property
    def cond_guid_weight(self) -> float:
        return self._config['cond_guid_weight']
    
    @cond_guid_weight.setter
    def cond_guid_weight(self, v: float) -> None:
        self._config['cond_guid_weight'] = v
        
    @property
    def cond_guid_threshold(self) -> float:
        return self._config['cond_guid_threshold']
    
    @cond_guid_threshold.setter
    def cond_guid_threshold(self, v: float) -> None:
        self._config['cond_guid_threshold'] = v
        
    @property
    def cond_momentum_weight(self) -> float:
        return self._config['cond_momentum_weight']
    
    @cond_momentum_weight.setter
    def cond_momentum_weight(self, v: float) -> None:
        self._config['cond_momentum_weight'] = v
    
    @property
    def cond_momentum_beta(self) -> float:
        return self._config['cond_momentum_beta']
    
    @cond_momentum_beta.setter
    def cond_momentum_beta(self, v: float) -> None:
        self._config['cond_momentum_beta'] = v
    
    @property
    def warmup_steps(self) -> int:
        return self._config['warmup_steps']
    
    @warmup_steps.setter
    def warmup_steps(self, v: int) -> None:
        self._config['warmup_steps'] = v
    
    @property
    def overall_guid_weight(self) -> float:
        return self._config['overall_guid_weight']
    
    @overall_guid_weight.setter
    def overall_guid_weight(self, v: float) -> None:
        self._config['overall_guid_weight'] = v
