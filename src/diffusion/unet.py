"""Modules for Unet and attention.

Summary:
    Unet composes of encoder and decoder.
    The encoder is a stack of downsampling blocks.
    The decoder is a stack of upsampling blocks.
    The encoder and decoder are connected by skip connections.
    The output of unet has the same size as the input.

Reference for Unet:
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion

Reference for attention:
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion
    - https://towardsdatascience.com/transformer-neural-network-step-by-step-breakdown-of-the-beast-b3e096dc857f
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class GeGLU(nn.Module):
    """GeGLU.
    
    GeGLU is an activation function commonly used in transformers.
    """
    
    def __init__(self, d_in: int, d_out: int) -> None:
        """Initialize.
        
        Args:
            d_in: the last dimension of input tensor.
            d_out: the last dimension of output tensor.
        """
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, ..., d_in]`.
        
        Returns:
            output tensor of shape `[batch_size, ..., d_out]`.
        """
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    """FeedForward.
    
    Feed-forward network is a multi-layer perceptron (MLP). Its main purpose 
    is to transform the attention vectors to a form that is acceptable by the next
    encoder or decoder layer.
    """
    
    def __init__(self, d_model: int, n_mult: int = 4) -> None:
        """Initialize.
        
        Args:
            d_model: the last dimension of input tensor to the feed-forward network.
            n_mult: the multiplication factor of the hidden layer dimension.
        """
        super().__init__()
        self.ffn = nn.Sequential(
            GeGLU(d_model, n_mult * d_model),
            nn.Dropout(0.),
            nn.Linear(n_mult * d_model, d_model),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, ..., d_model]`.
        
        Returns:
            output tensor of shape `[batch_size, ..., d_model]`.
        """
        return self.ffn(x)

class CrossAttention(nn.Module):
    """CrossAttention.
        
    Attention is the core component of the transformer architecture.
    """

    def __init__(self, d_model: int, d_cond_emb: int, n_heads: int, d_head: int, is_inplace: bool = True) -> None:
        """Initialize.
        
        Args:
            d_model: the last dimension of input tensor and the output tensor.
            d_cond_emb: the last dimension of cond_embition tensor.
            n_heads: the number of attention heads.
            d_head: the dimension of each attention head.
            is_inplace: whether to use softmax inplace to save memory.
        """
        super().__init__()
        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        
        # project the input tensor to the query, key, and value tensors for all heads at once
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond_emb, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond_emb, d_attn, bias=False)
        
        # project mutli-head attention back to the original input dimension
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor = None) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, d_model]`.
            cond_emb: cond_embition tensor of shape `[batch_size, n_cond_emb, d_cond_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, d_model]`.
        """
        assert len(x.shape) == 3, f'input tensor must be of shape [batch_size, n_channels, d_model], but got {x.shape}'
        
        # if no cond_embition is provided, perform self-attention
        has_cond_emb = cond_emb is not None
        if not has_cond_emb:
            cond_emb = x
        else:
            x_size, cond_emb_size = x.shape[0], cond_emb.shape[0]
            assert len(cond_emb.shape) == 3, f'cond_embition tensor must be of shape [batch_size, n_cond_emb, d_cond_emb], but got {cond_emb.shape}'
            assert x_size == cond_emb_size, f'batch size of input tensor and cond_embition tensor must be the same, but got {x_size} and {cond_emb_size}'
        
        # project the input tensor to the query, key, and value tensors for all heads at once
        q = self.to_q(x)
        k = self.to_k(cond_emb)
        v = self.to_v(cond_emb)
        
        return self.attention(q, k, v)
    
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Perform attention.
        
        Args:
            q: query tensor of shape `[batch_size, n_channels, d_attn]`.
            k: key tensor of shape `[batch_size, n_cond_emb, d_attn]`.
            v: value tensor of shape `[batch_size, n_cond_emb, d_attn]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, d_model]`.
        """
        # split them to heads of shape `[batch_size, n_channels, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # calculate attention $\frac{Q K^\top}{\sqrt{d}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        
        # compute attention output with ${softmax}(\frac{Q K^\top}{\sqrt{d}})V$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)

        # reshape to `[batch_size, n_channels, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        
        # map to `[batch_size, n_channels, d_model]` with a linear layer
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    """BasicTransformerBlock.
    
    A basic transformer block consists of a multi-head attention layer and a feed-forward network.
    """
    
    def __init__(self, d_model: int, d_cond_emb: int, n_heads: int, d_head: int) -> None:
        """Initialize.
        
        Args:
            d_model: the last dimension of input tensor and the output tensor.
            d_cond_emb: the last dimension of cond_embition tensor.
            n_heads: the number of attention heads.
            d_head: the dimension of each attention head.
        """
        super().__init__()
        # pre-norm layer and self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = CrossAttention(d_model, d_model, n_heads, d_head)
        # pre-norm layer and cross attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, d_cond_emb, n_heads, d_head)
        # pre-norm layer and feed-forward network
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
    
    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of shape `[batch_size, n_channels, d_model]`.
            cond_emb: cond_embition tensor of shape `[batch_size, n_cond_emb, d_cond_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, d_model]`.
        """
        # self-attention and skip connection
        x = self.self_attn(self.norm1(x)) + x
        # cross attention and skip connection
        x = self.cross_attn(self.norm2(x), cond_emb=cond_emb) + x
        # feed-forward network and skip connection
        return self.ffn(self.norm3(x)) + x
        
class SpatialTransformer(nn.Module):
    """SpatialTransformer."""
    
    def __init__(
        self, n_channels: int, n_layers: int, n_heads: int, d_cond_emb: int,
        n_groups: int = 32, eps: float = 1e-6,
    ) -> None:
        """Init.
        
        Args:
            n_channels: the number of channels of input tensor.
            n_layers: the number of transformer blocks.
            n_heads: the number of attention heads.
            d_cond_emb: the last dimension of cond_embition tensor.
            n_groups: the number of groups for group normalization.
            eps: a value added to the denominator for numerical stability in group normalization.
        """
        super().__init__()
        # initial group normalization
        assert n_channels % n_groups == 0, f'`n_channels` must be divisible by `n_groups`, but got {n_channels} and {n_groups}'
        self.group_norm = nn.GroupNorm(n_groups, n_channels, eps=eps, affine=True)
        # initial input projection
        self.proj_in = nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=1, padding=0)
        # transformer blocks
        transformer_blocks = []
        # partition the `d_model` uniformly in attention heads by setting `d_head = d_model // n_heads`
        d_head = n_channels // n_heads
        assert d_head > 0, f'`d_head` must be greater than 0, but got {d_head}'
        for _ in range(n_layers):
            transformer_blocks.append(
                BasicTransformerBlock(
                    d_model=n_channels, d_cond_emb=d_cond_emb, n_heads=n_heads, d_head=d_head,
                ),
            )
        self.transformer_blocks = nn.ModuleList(transformer_blocks)
        self.proj_out = nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            x: input tensor of chape `[batch_size, n_channels, height, width]`.
            cond_emb: cond_embition tensor of shape `[batch_size, n_cond_emb, d_cond_emb]`.
        
        Returns:
            output tensor of shape `[batch_size, n_channels, height, width]`.
        """
        # get shape of input tensor `[batch_size, n_channels, height, width]`
        b, c, h, w = x.shape
        # for residual connection
        x_in = x
        # group normalization among channels
        x = self.group_norm(x)
        # print(f'x.shape: {list(x.shape)} after `GroupNorm`')
        # initial $1 \times 1$ convolution
        x = self.proj_in(x)
        # print(f'x.shape: {list(x.shape)} after `Conv2d` with kernel size of 1')
        # transpose and reshape from `[batch_size, n_channels, height, width]` to `[batch_size, height * width, n_channels]`
        # NOTE: `height * width` will be `n_channels` in `BasicTransformerBlock`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # print(f'x.shape: {list(x.shape)} after `permute` and `view`')
        # apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, cond_emb)
            # print(f'x.shape: {list(x.shape)} after `BasicTransformerBlock`')
        # reshape and transpose from `[batch_size, height * width, n_channels]` to `[batch_size, n_channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # print(f'x.shape: {list(x.shape)} after `view` and `permute`')
        # final $1 \times 1$ convolution and residual connection
        x = self.proj_out(x) + x_in
        # print(f'x.shape: {list(x.shape)} after `Conv2d` with kernel size of 1 and skip connection')
        return x

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
    
    print('---------------------------------------- test1 ----------------------------------------')
    # simple input
    x = torch.randn(1, 1, 3)
    print(f'x: {x}')
    x = GeGLU(d_in=3, d_out=3)(x)
    print(f'x: {x} after `GeGLU`')
    x = FeedForward(d_model=3)(x)
    print(f'x: {x} after `FeedForward`')
    cond_emb = torch.randn(1, 1, 3)
    # `d_model` equals to last dimension of `x`, `d_cond_emb` equals to last dimension of `cond_emb`
    x = CrossAttention(d_model=3, d_cond_emb=3, n_heads=2, d_head=3)(x, cond_emb)
    print(f'x: {x} after `CrossAttention`')
    x = BasicTransformerBlock(d_model=3, d_cond_emb=3, n_heads=2, d_head=3)(x, cond_emb)
    print(f'x: {x} after `BasicTransformerBlock`')

    print('\n---------------------------------------- test2 ----------------------------------------')
    # batches
    n_batches = 10
    n_channels = 32
    height, width = 3, 3
    x = torch.randn(n_batches, n_channels, height, width)
    cond_emb = torch.randn(n_batches, 2, 3)
    print(f'x.shape: {list(x.shape)} before `SpatialTransformer`')
    x = SpatialTransformer(
        n_channels, n_layers=2, n_heads=10, d_cond_emb=3,
    )(x, cond_emb)
    print(f'x.shape: {list(x.shape)} after `SpatialTransformer`')
    
    # simple input
    print('\n---------------------------------------- test3 ----------------------------------------')
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
    
    print('\n---------------------------------------- test4 ----------------------------------------')
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
