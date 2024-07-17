"""Modules for Unet.

Summary:
    Unet composes of encoder and decoder.
    The encoder is a stack of downsampling blocks.
    The decoder is a stack of upsampling blocks.
    The encoder and decoder are connected by skip connections.
    The output of unet has the same size as the input.

Reference:
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from attention import SpatialTransformer

class DownSample(nn.Module):
    """DownSample."""
    
    def __init__(self, n_channels: int) -> None:
        """Initialize.
        
        Args:
            n_channels: number of channels for input and output.
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.down = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, height, width]`.
            
        Returns:
            output tensor of shape `[batch_size, n_channels, ceil(height // 2), ceil(width // 2)]`.
        """
        return self.down(x)

class UpSample(nn.Module):
    """UpSample."""
    
    def __init__(self, n_channels: int) -> None:
        """Initialize.
        
        Args:
            n_channels: number of channels for input and output.
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, height, width]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, height * 2, width * 2]`.
        """
        # up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # apply convolution
        return self.conv(x)

class GroupNorm(nn.GroupNorm):
    """GroupNorm.
    
    Group Normalization is a technique used in deep learning to normalize the activations of a 
    neural network layer across a group of channels instead of normalizing across the entire batch. 
    This can be useful when the batch size is small or when the channels are highly correlated.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, ...]`.

        Returns:
            output tensor of shape `[batch_size, n_channels, ...]`.
        """
        # first convert to float and then convert back to the original type
        return super().forward(x.float()).type(x.dtype)

def group_norm(n_channels: int, n_groups: int = 32) -> nn.Module:
    """GroupNorm.
    
    Raises:
        ValueError: if `n_channels` is not divisible by `n_groups`.
    
    Args:
        n_channels (int): number of channels.
        n_groups (int): number of groups.
    
    Returns:
        GroupNorm module.
    """
    try:
        return GroupNorm(n_groups, n_channels)
    except ValueError as e:
        print(f'n_channels: {n_channels}, n_groups: {n_groups}')
        raise e

class ResBlock(nn.Module):
    """ResBlock."""
    
    def __init__(
        self, 
        n_in_channels: int, 
        d_t_emb: int, 
        n_out_channels: int = None, 
        n_groups: int = 32,
    ) -> None:
        """Initialize.
        
        Args:
            n_in_channels: number of input channels.
            d_t_emb: dimension of timestep embedding.
            n_out_channels: number of output channels.
            n_groups: number of groups for group normalization.
        """
        super().__init__()
        # default number of output channels is the same as the number of input channels
        if n_out_channels is None:
            n_out_channels = n_in_channels
        
        # initial convolution
        # `stride=1`, `padding=1`, so the output of convolution has the same size as the input
        self.in_layers = nn.Sequential(
            group_norm(n_in_channels, n_groups=n_groups),
            nn.SiLU(),
            nn.Conv2d(n_in_channels, n_out_channels, kernel_size=3, stride=1, padding=1),
        )
        
        # time embedding layers
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, n_out_channels),
        )
        
        # final convolution
        self.out_layers = nn.Sequential(
            group_norm(n_out_channels, n_groups=n_groups),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding=1),
        )
        
        # skip connection
        if n_out_channels == n_in_channels:
            self.skip_connection = nn.Identity()
        else:
            # `kernel_size=1`, `stride=1`, so the output of convolution is the same as the input
            self.skip_connection = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1, stride=1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, height, width]`.
            t_emb: timestep embedding tensor of shape `[batch_size, d_t_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, height, width]`.
        """
        # initial convolution (`h` has the same shape as `x`)
        h = self.in_layers(x)
        # print(f'h.shape: {list(h.shape)} after initial convolution')
        
        # timestep embeddings
        # `t_emb` has the same shape as `h` for the first two dimensions
        t_emb = self.t_emb_layers(t_emb).type(h.dtype)
        # print(f't_emb.shape: {list(t_emb.shape)}')
        
        # add time embedding to the output of the initial convolution
        h = h + t_emb[:, :, None, None]
        # print(f'h.shape: {list(h.shape)} after adding timestep embedding')
        
        # final convolution (`h` has the same shape as `x`)
        h = self.out_layers(h)
        # print(f'h.shape: {list(h.shape)} after final convolution')
        
        # skip connection
        x = self.skip_connection(x) + h
        # print(f'x.shape: {list(x.shape)} after skip connection')
        return x
    
class TimestepEmbedSequential(nn.Sequential):
    """TimestepEmbedSequential.
    
    This sequential module can compose of different modules such as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures.
    """
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor = None) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, height, width]`.
            t_emb: timestep embedding tensor of shape `[batch_size, d_t_emb]`.
            cond_emb: cond_embition tensor of shape `[batch_size, n_cond_emb, d_cond_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, height, width]`.
        """
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond_emb)
            else:
                x = layer(x)
        return x

class Unet(nn.Module):
    """Unet."""
    
    def __init__(
        self, 
        n_in_channels: int,
        n_out_channels: int,
        n_base_channels: int,
        n_channels_factors: List[int],
        n_res_blocks: int,
        attention_levels: List[int],
        d_t_emb: int,
        d_cond_emb: int,
        n_tf_layers: int = 1,
        n_heads: int = 1,
        n_groups: int = 1,
    ) -> None:
        """Initialize.
        
        Args:
            n_in_channels: number of input channels.
            n_out_channels: number of output channels.
            n_base_channels: number of base channels.
            n_channels_factors: list of factors of `n_base_channels` at each level.
            n_res_blocks: number of residual blocks at each level.
            attention_levels: list of levels where `SpatialTransformer` is added.
            d_t_emb: dimension of timestep embedding.
            d_cond_emb: dimension of cond_embition embedding.
            n_tf_layers: number of layers in `SpatialTransformer`.
            n_heads: number of heads in `SpatialTransformer`.
            n_groups: number of groups for group normalization.
        """
        super().__init__()
        # record the arguments
        self.n_in_channels = n_in_channels
        self.n_levels = len(n_channels_factors)
        
        # downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            TimestepEmbedSequential(
                # initial convolution that maps `n_in_channels` to `n_base_channels`
                # `stride=1`, `padding=1`, so the output of convolution has the same size as the input
                nn.Conv2d(n_in_channels, n_base_channels, kernel_size=3, stride=1, padding=1),
            ),
        )
        
        # number of levels
        n_levels = len(n_channels_factors)
        # print(f'n_levels: {n_levels}')
        
        # number of channels at each block of downsampling path
        n_channels_at_down_blocks = [n_base_channels]
        
        # number of channels at each level
        n_channels_at_levels = [n_base_channels * factor for factor in n_channels_factors]
        
        # downsampling blocks of all levels
        for down_level in range(n_levels):
            # list of [`ResBlock`, `SpatialTransformer`] if `down_level` is in `attention_levels`
            # list of [`ResBlock`] otherwise
            for _ in range(n_res_blocks):
                down_layers = [
                    ResBlock(
                        n_in_channels=n_base_channels,
                        d_t_emb=d_t_emb,
                        n_out_channels=n_channels_at_levels[down_level],
                        n_groups=n_groups,
                    ),
                ]
                # update `n_base_channels` as `n_out_channels` of the last `ResBlock`
                # `n_base_channels` is used as the number of input channels at the next level
                n_base_channels = n_channels_at_levels[down_level]
                # add `SpatialTransformer` if `down_level` is in `attention_levels`
                if down_level in attention_levels:
                    down_layers.append(
                        SpatialTransformer(
                            n_channels=n_base_channels,
                            n_layers=n_tf_layers,
                            n_heads=n_heads,
                            n_groups=n_groups,
                            d_cond_emb=d_cond_emb,
                        ),
                    )
                self.down_blocks.append(TimestepEmbedSequential(*down_layers))
                n_channels_at_down_blocks.append(n_base_channels)
            # downsample at all levels except the last level
            if down_level != n_levels - 1:
                self.down_blocks.append(TimestepEmbedSequential(DownSample(n_base_channels)))
                n_channels_at_down_blocks.append(n_base_channels)
        
        # print(f'n_channels_at_down_blocks: {n_channels_at_down_blocks}')
        # print(f'n_channels_at_levels: {n_channels_at_levels}')
        
        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(n_in_channels=n_base_channels, d_t_emb=d_t_emb, n_groups=n_groups),
            SpatialTransformer(
                n_channels=n_base_channels, n_heads=n_heads, n_layers=n_tf_layers, 
                d_cond_emb=d_cond_emb, n_groups=n_groups,
            ),
            ResBlock(n_in_channels=n_base_channels, d_t_emb=d_t_emb, n_groups=n_groups),
        )
        
        # upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        # upsampling blocks of all levels
        for up_level in reversed(range(n_levels)):
            for res_block_idx in range(n_res_blocks + 1):
                up_layers = [
                    ResBlock(
                        # add skip connection from the corresponding downsampling block in channel dimension
                        n_in_channels=n_base_channels + n_channels_at_down_blocks.pop(),
                        d_t_emb=d_t_emb,
                        n_out_channels=n_channels_at_levels[up_level],
                        n_groups=n_groups,
                    ),
                ]
                # update `n_base_channels` as `n_out_channels` of the last `ResBlock`
                n_base_channels = n_channels_at_levels[up_level]
                # add `SpatialTransformer` if `up_level` is in `attention_levels`
                if up_level in attention_levels:
                    up_layers.append(
                        SpatialTransformer(
                            n_channels=n_base_channels,
                            n_layers=n_tf_layers,
                            n_heads=n_heads,
                            n_groups=n_groups,
                            d_cond_emb=d_cond_emb,
                        ),
                    )
                # upsample at all levels except the last level
                if up_level != 0 and res_block_idx == n_res_blocks:
                    up_layers.append(UpSample(n_base_channels))
                # add upsampling block
                self.up_blocks.append(TimestepEmbedSequential(*up_layers))

        # final normalization and convolution
        self.out_layers = nn.Sequential(
            group_norm(n_base_channels, n_groups=n_groups),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(n_base_channels, n_out_channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, height, width]`.
            t_emb: timestep embedding tensor of shape `[batch_size, d_t_emb]`.
            cond_emb: cond_embition tensor of shape `[batch_size, n_cond_emb, d_cond_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, height, width]`.
        """
        min_height, min_width = 2 ** (self.n_levels - 1), 2 ** (self.n_levels - 1)
        assert x.shape[0] == t_emb.shape[0] == cond_emb.shape[0], 'batch size mismatch'
        assert len(x.shape) == 4, 'input tensor must be 4-dimensional in the shape of [batch_size, n_channels, height, width]'
        assert x.shape[1] == self.n_in_channels, 'number of input channels mismatch'
        assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, 'height and width of input tensor must be even'
        assert x.shape[2] >= min_height and x.shape[3] >= min_width, f'input tensor should be at least {min_height}x{min_width} given {self.n_levels} levels'
        assert len(t_emb.shape) == 2, 'timestep embedding tensor must be 2-dimensional in the shape of [batch_size, d_t_emb]'
        assert len(cond_emb.shape) == 3, 'cond_embition tensor must be 3-dimensional in the shape of [batch_size, n_cond_emb, d_cond_emb]'
        # store the intermediate outputs of downsampling blocks for skip connections
        x_down_blocks = []
        
        # downsampling blocks
        # print(f'x.shape: {list(x.shape)} before `Unet`')
        for down_block in self.down_blocks:
            x = down_block(x, t_emb, cond_emb)
            x_down_blocks.append(x)
            # print(f'x.shape: {list(x.shape)} after downsampling block')

        # middle block
        x = self.middle_block(x, t_emb, cond_emb)
        # print(f'x.shape: {list(x.shape)} after middle block')
        
        # upsampling blocks
        for up_block in self.up_blocks:
            # pop the intermediate output of the corresponding downsampling block for skip connection
            x_skip = x_down_blocks.pop()
            # print(f'x_skip.shape: {list(x_skip.shape)}')
            # print(f'x.shape: {list(x.shape)}')
            # concatenate the intermediate output of the corresponding downsampling block
            # to the output of the previous upsampling block at channel dimension
            x = torch.cat([x, x_skip], dim=1)
            # print(f'x.shape: {list(x.shape)} after concatenation')
            x = up_block(x, t_emb, cond_emb)
            # print(f'x.shape: {list(x.shape)} after upsampling block')

        # final normalization and convolution
        x = self.out_layers(x)
        # print(f'x.shape: {list(x.shape)} after `Unet`')
        return x

def _test():
    # random seed
    torch.manual_seed(0)
    
    # simple input
    print('---------------------------------------- test1 ----------------------------------------')
    batch_size = 1
    n_channels = 1
    d_t_emb = 10
    d_cond_emb = 8
    n_groups = 1
    x = torch.randn(batch_size, n_channels, 2, 2)
    cond_emb = torch.randn(batch_size, 1, d_cond_emb)
    t_emb = torch.randn(batch_size, d_t_emb)
    print(f'x: {x}')
    x = DownSample(n_channels=n_channels)(x)
    print(f'x: {x} after `DownSample`')
    x = UpSample(n_channels=n_channels)(x)
    print(f'x: {x} after `UpSample`')
    x = group_norm(n_channels=n_channels, n_groups=n_groups)(x)
    print(f'x: {x} after `GroupNorm`')
    x = ResBlock(
        n_in_channels=n_channels, d_t_emb=d_t_emb, n_groups=n_groups,
    )(x, t_emb)
    print(f'x: {x} after `ResBlock`')
    x = Unet(
        n_in_channels=n_channels,
        n_out_channels=n_channels,
        n_base_channels=1,
        n_channels_factors=[3, 4],
        n_res_blocks=1,
        attention_levels=[0],
        d_t_emb=d_t_emb,
        d_cond_emb=d_cond_emb,
        n_groups=n_groups,
    )(x, t_emb, cond_emb)
    print(f'x: {x} after `Unet`')
    
    print('\n---------------------------------------- test2 ----------------------------------------')
    # high-level test
    batch_size = 5
    in_height = 8
    in_width = 8
    d_t_emb = 7
    n_cond_emb = 1
    d_cond_emb = 8
    n_groups = 1
    test_in = torch.randn(batch_size, 1, in_height, in_width)
    t_emb = torch.randn(batch_size, d_t_emb)
    cond_emb = torch.randn(batch_size, n_cond_emb, d_cond_emb)
    test_out = Unet(
        n_in_channels=1,
        n_out_channels=1,
        n_base_channels=1,
        n_channels_factors=[2, 4, 4, 3],
        n_res_blocks=1,
        attention_levels=[0],
        d_t_emb=d_t_emb,
        d_cond_emb=d_cond_emb,
        n_groups=n_groups,
    )(test_in, t_emb, cond_emb)
    print(f'x.shape: {list(test_in.shape)} before `Unet`')
    print(f'x.shape: {list(test_out.shape)} after `Unet`')


if __name__ == '__main__':
    _test()
