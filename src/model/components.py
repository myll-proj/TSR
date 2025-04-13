from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation

import copy
from typing import Optional, Union, Callable
import math
import torch.nn.functional as F  # Add this line
from torch.nn import MultiheadAttention
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm


__all__ = [
    "ImgCnnBackbone",
    "ImgLinearBackbone",
    "ImgConvStemBackbone",
    "PositionEmbedding",
    "Encoder",
    "Decoder",
    "TokenEmbedding",
]


class ImgCnnBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_channels: int,
        d_model: int,
        drop_layer: Tuple = None,
    ) -> None:
        super().__init__()

        # drop layers for classification & maxpooling for higher feature resolution
        layers = list(backbone.children())
        nlayer = len(layers)
        keep_layer = set([i for i in range(nlayer)]) - set(drop_layer)
        backbone = [layers[i] for i in keep_layer]
        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(output_channels, d_model)
        self.channels = output_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        assert x.shape[-1] == self.channels, "Image channels size mismatch."
        x = self.proj(x)
        return x


class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_chan: int = 3,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_chan, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class ImgConvStemBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        downsample_factor: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0
        assert output_channels % (downsample_factor // 2) == 0
        input_channels = output_channels // (downsample_factor // 2)

        layers = [
            Conv2dNormActivation(
                3, input_channels, kernel_size=kernel_size, stride=2, padding=1
            )
        ]

        while input_channels != output_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    input_channels * 2,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            input_channels = input_channels * 2

        layers.append(nn.Conv2d(output_channels, d_model, kernel_size=1))

        self.conv_stem = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, nlayer)

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        x = self.decoder(
            x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask
        )
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # assume x is batch first
        out = self.embedding(torch.arange(x.shape[1], device=x.device))
        return self.dropout(out + x)


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        assert vocab_size > 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class GatedGLU(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(GatedGLU, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        output = torch.sigmoid(self.linear1(x))
        gate = F.gelu(output)
        combined = gate * self.linear2(x)
        return self.linear3(combined)


class my_TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(my_TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)

        self.mlp = GatedGLU(d_model, dim_feedforward)

        self.norm_first = norm_first

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(my_TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self.mlp(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = x + self.mlp(self.norm3(x))

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        return x


class PrintLayer(nn.Module):
    """Only for debugging when loss is nan."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(
            "torch.isfinite(x).all(): {}, min. {:.5f}, max. {:.5f}".format(
                torch.isfinite(x).all(), x.min(), x.max()
            )
        )
        return x


if __name__ == "__main__":
    from torchvision import models

    x = torch.rand(1, 3, 392, 392)
    model = ImgConvStemBackbone(
        d_model=512, downsample_factor=16, output_channels=64, kernel_size=5
    )
    y = model(x)
    print(model)
    print(y.shape)

    model = ImgCnnBackbone(
        backbone=models.resnet34(),
        output_channels=512,
        d_model=512,
        drop_layer=(3, 8, 9),
    )

    # print(model)
    y = model(x)
    print(y.shape)
