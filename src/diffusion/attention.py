"""Modules for attention mechanism and transformer blocks.

Reference:
    - https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion
    - https://towardsdatascience.com/transformer-neural-network-step-by-step-breakdown-of-the-beast-b3e096dc857f
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == '__main__':
    _test()
