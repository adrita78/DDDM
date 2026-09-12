from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from .attention import SpatialTransformer


@th.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.

    Warning:
    torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        th.ones_like(tensor)
        for _ in range(th.distributed.get_world_size())
    ]

    th.distributed.all_gather(
        tensors_gather,
        tensor,
        async_op=False,
    )

    output = th.cat(tensors_gather, dim=0)

    return output


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given a timestep embedding.
        """
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to children
    that support it as an extra input.

    Added `.contiguous()` calls to make this robust to tensors generated
    by torch.func.jvp/vjp.
    """

    def forward(self, x, emb, context=None):

        x = x.contiguous()

        for layer in self:

            x = x.contiguous()

            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)

            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)

            else:
                x = layer(x)

        return x.contiguous()


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()

        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        if use_conv:
            self.conv = conv_nd(
                dims,
                channels,
                channels,
                3,
                padding=1,
            )

    def forward(self, x):

        x = x.contiguous()

        assert x.shape[1] == self.channels

        if self.dims == 3:
            x = F.interpolate(
                x,
                (
                    x.shape[2],
                    x.shape[3] * 2,
                    x.shape[4] * 2,
                ),
                mode="nearest",
            )
        else:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="nearest",
            )

        x = x.contiguous()

        if self.use_conv:
            x = self.conv(x)

        return x.contiguous()


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    """

    def __init__(self, channels, use_conv, dims=2):

        super().__init__()

        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.op = conv_nd(
                dims,
                channels,
                channels,
                3,
                stride=stride,
                padding=1,
            )
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):

        x = x.contiguous()

        assert x.shape[1] == self.channels

        x = self.op(x)

        return x.contiguous()


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()

        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(
                dims,
                channels,
                self.out_channels,
                3,
                padding=1,
            ),
        )

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                (
                    2 * self.out_channels
                    if use_scale_shift_norm
                    else self.out_channels
                ),
            ),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                )
            ),
        )

        if self.out_channels == channels:

            self.skip_connection = nn.Identity()

        elif use_conv:

            self.skip_connection = conv_nd(
                dims,
                channels,
                self.out_channels,
                3,
                padding=1,
            )

        else:

            self.skip_connection = conv_nd(
                dims,
                channels,
                self.out_channels,
                1,
            )

    def forward(self, x, emb):

        x = x.contiguous()
        emb = emb.contiguous()

        return checkpoint(
            self._forward,
            (x, emb),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, emb):

        x = x.contiguous()
        emb = emb.contiguous()

        h = self.in_layers(x)

        h = h.contiguous()

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        emb_out = emb_out.contiguous()

        if self.use_scale_shift_norm:

            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]

            scale, shift = th.chunk(
                emb_out,
                2,
                dim=1,
            )

            scale = scale.contiguous()
            shift = shift.contiguous()

            h = out_norm(h.contiguous())

            h = h * (1 + scale) + shift

            h = h.contiguous()

            h = out_rest(h)

        else:

            h = h + emb_out

            h = h.contiguous()

            h = self.out_layers(h)

        h = h.contiguous()

        skip = self.skip_connection(
            x.contiguous()
        )

        return (
            skip + h
        ).contiguous()


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Added `.contiguous()` calls to make this robust to tensors produced
    by torch.func.jvp/vjp.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        use_checkpoint=False,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)

        self.qkv = conv_nd(
            1,
            channels,
            channels * 3,
            1,
        )

        self.attention = QKVAttention()

        self.proj_out = zero_module(
            conv_nd(
                1,
                channels,
                channels,
                1,
            )
        )

    def forward(self, x):

        x = x.contiguous()

        return checkpoint(
            self._forward,
            (x,),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x):

        x = x.contiguous()

        b, c, *spatial = x.shape

        # Flatten spatial dimensions.
        x = x.reshape(
            b,
            c,
            -1,
        ).contiguous()

        # IMPORTANT:
        # GroupNorm can fail on non-contiguous tensors produced
        # by torch.func.jvp/vjp.
        x_norm = self.norm(
            x.contiguous()
        ).contiguous()

        qkv = self.qkv(
            x_norm
        ).contiguous()

        qkv = qkv.reshape(
            b * self.num_heads,
            -1,
            qkv.shape[2],
        ).contiguous()

        h = self.attention(
            qkv
        ).contiguous()

        h = h.reshape(
            b,
            -1,
            h.shape[-1],
        ).contiguous()

        h = self.proj_out(
            h.contiguous()
        ).contiguous()

        output = (
            x + h
        ).reshape(
            b,
            c,
            *spatial,
        ).contiguous()

        return output


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):

        qkv = qkv.contiguous()

        ch = qkv.shape[1] // 3

        q, k, v = th.split(
            qkv,
            ch,
            dim=1,
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        scale = 1 / math.sqrt(
            math.sqrt(ch)
        )

        weight = th.einsum(
            "bct,bcs->bts",
            q * scale,
            k * scale,
        )

        weight = th.softmax(
            weight.float(),
            dim=-1,
        ).type(
            weight.dtype
        ).contiguous()

        output = th.einsum(
            "bts,bcs->bct",
            weight,
            v,
        )

        return output.contiguous()

    @staticmethod
    def count_flops(model, _x, y):

        b, c, *spatial = y[0].shape

        num_spatial = int(
            np.prod(spatial)
        )

        matmul_ops = (
            2
            * b
            * (num_spatial ** 2)
            * c
        )

        model.total_ops += th.DoubleTensor(
            [matmul_ops]
        )


class UNetModel(nn.Module):
    """
    The full UNet model.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_CA=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        self.register_buffer(
            "x_bar",
            th.randn(
                50000,
                3,
                32,
                32,
            ),
        )

        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(
                model_channels,
                time_embed_dim,
            ),
            SiLU(),
            linear(
                time_embed_dim,
                time_embed_dim,
            ),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(
                num_classes,
                time_embed_dim,
            )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims,
                        in_channels,
                        model_channels,
                        3,
                        padding=1,
                    )
                )
            ]
        )

        input_block_chans = [
            model_channels
        ]

        ch = model_channels
        ds = 1

        for level, mult in enumerate(
            channel_mult
        ):

            for _ in range(
                num_res_blocks
            ):

                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=(
                            mult * model_channels
                        ),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=(
                            use_scale_shift_norm
                        ),
                    )
                ]

                ch = mult * model_channels

                if ds in attention_resolutions:

                    if not use_CA:

                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                            )
                        )

                    else:

                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                time_embed_dim // num_heads,
                                context_dim=time_embed_dim,
                            )
                        )

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        *layers
                    )
                )

                input_block_chans.append(
                    ch
                )

            if level != len(channel_mult) - 1:

                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                        )
                    )
                )

                input_block_chans.append(
                    ch
                )

                ds *= 2

        if not use_CA:

            middle_attention = AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
            )

        else:

            middle_attention = SpatialTransformer(
                ch,
                num_heads,
                time_embed_dim // num_heads,
                context_dim=time_embed_dim,
            )

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=(
                    use_scale_shift_norm
                ),
            ),
            middle_attention,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=(
                    use_scale_shift_norm
                ),
            ),
        )

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(
            enumerate(channel_mult)
        )[::-1]:

            for i in range(
                num_res_blocks + 1
            ):

                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=(
                            model_channels * mult
                        ),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=(
                            use_scale_shift_norm
                        ),
                    )
                ]

                ch = model_channels * mult

                if ds in attention_resolutions:

                    if not use_CA:

                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                            )
                        )

                    else:

                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                time_embed_dim // num_heads,
                                context_dim=time_embed_dim,
                            )
                        )

                if level and i == num_res_blocks:

                    layers.append(
                        Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                        )
                    )

                    ds //= 2

                self.output_blocks.append(
                    TimestepEmbedSequential(
                        *layers
                    )
                )

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(
                conv_nd(
                    dims,
                    model_channels,
                    out_channels,
                    3,
                    padding=1,
                )
            ),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """

        self.input_blocks.apply(
            convert_module_to_f16
        )

        self.middle_block.apply(
            convert_module_to_f16
        )

        self.output_blocks.apply(
            convert_module_to_f16
        )

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """

        self.input_blocks.apply(
            convert_module_to_f32
        )

        self.middle_block.apply(
            convert_module_to_f32
        )

        self.output_blocks.apply(
            convert_module_to_f32
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """

        return next(
            self.input_blocks.parameters()
        ).dtype

    def forward(
        self,
        x,
        timesteps,
        context=None,
        y=None,
    ):
        """
        Apply the model to an input batch.

        :param x:
            an [N x C x ...] Tensor of inputs.

        :param timesteps:
            a 1-D batch of timesteps.

        :param context:
            Optional conditioning context.

        :param y:
            an [N] Tensor of labels, if class-conditional.

        :return:
            an [N x C x ...] Tensor of outputs.
        """

        assert (
            (y is not None)
            == (
                self.num_classes is not None
            )
        ), (
            "must specify y if and only if "
            "the model is class-conditional"
        )

        hs = []

        emb = self.time_embed(
            timestep_embedding(
                timesteps,
                self.model_channels,
            )
        )

        emb = emb.contiguous()

        if self.num_classes is not None:

            assert y.shape == (
                x.shape[0],
            )

            emb = (
                emb
                + self.label_emb(y)
            ).contiguous()

        # IMPORTANT:
        # torch.func.jvp/vjp can produce tensors with
        # unusual stride layouts.
        h = x.type(
            self.inner_dtype
        ).contiguous()

        # -------------------------
        # Input / downsampling path
        # -------------------------

        for module in self.input_blocks:

            h = h.contiguous()

            h = module(
                h,
                emb,
                context,
            )

            h = h.contiguous()

            hs.append(h)

        # -------------------------
        # Middle block
        # -------------------------

        h = h.contiguous()

        h = self.middle_block(
            h,
            emb,
            context,
        )

        h = h.contiguous()

        # -------------------------
        # Output / upsampling path
        # -------------------------

        for module in self.output_blocks:

            skip = hs.pop().contiguous()

            h = h.contiguous()

            cat_in = th.cat(
                [
                    h,
                    skip,
                ],
                dim=1,
            ).contiguous()

            h = module(
                cat_in,
                emb,
                context,
            )

            h = h.contiguous()

        h = h.type(
            x.dtype
        ).contiguous()

        h = self.out(
            h.contiguous()
        )

        return h.contiguous()

    def get_feature_vectors(
        self,
        x,
        timesteps,
        y=None,
    ):
        """
        Apply the model and return all of the intermediate tensors.
        """

        hs = []

        emb = self.time_embed(
            timestep_embedding(
                timesteps,
                self.model_channels,
            )
        )

        emb = emb.contiguous()

        if self.num_classes is not None:

            assert y.shape == (
                x.shape[0],
            )

            emb = (
                emb
                + self.label_emb(y)
            ).contiguous()

        result = dict(
            down=[],
            up=[],
        )

        h = x.type(
            self.inner_dtype
        ).contiguous()

        for module in self.input_blocks:

            h = h.contiguous()

            h = module(
                h,
                emb,
            )

            h = h.contiguous()

            hs.append(h)

            result["down"].append(
                h.type(
                    x.dtype
                ).contiguous()
            )

        h = h.contiguous()

        h = self.middle_block(
            h,
            emb,
        )

        h = h.contiguous()

        result["middle"] = (
            h.type(
                x.dtype
            ).contiguous()
        )

        for module in self.output_blocks:

            skip = hs.pop().contiguous()

            cat_in = th.cat(
                [
                    h,
                    skip,
                ],
                dim=1,
            ).contiguous()

            h = module(
                cat_in,
                emb,
            )

            h = h.contiguous()

            result["up"].append(
                h.type(
                    x.dtype
                ).contiguous()
            )

        return result

    @th.no_grad()
    def update_xbar(
        self,
        x,
        indices,
    ):

        data = concat_all_gather(
            x
        )

        indices = concat_all_gather(
            indices
        )

        bz = data.shape[0]

        assert indices.shape[0] == bz

        self.x_bar[
            indices
        ] = data


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    """

    def __init__(
        self,
        in_channels,
        *args,
        **kwargs,
    ):

        super().__init__(
            in_channels * 2,
            *args,
            **kwargs,
        )

    def forward(
        self,
        x,
        timesteps,
        low_res=None,
        **kwargs,
    ):

        _, _, new_height, new_width = x.shape

        upsampled = F.interpolate(
            low_res,
            (
                new_height,
                new_width,
            ),
            mode="bilinear",
        )

        x = th.cat(
            [
                x,
                upsampled,
            ],
            dim=1,
        ).contiguous()

        return super().forward(
            x,
            timesteps,
            **kwargs,
        )

    def get_feature_vectors(
        self,
        x,
        timesteps,
        low_res=None,
        **kwargs,
    ):

        _, new_height, new_width, _ = x.shape

        upsampled = F.interpolate(
            low_res,
            (
                new_height,
                new_width,
            ),
            mode="bilinear",
        )

        x = th.cat(
            [
                x,
                upsampled,
            ],
            dim=1,
        ).contiguous()

        return super().get_feature_vectors(
            x,
            timesteps,
            **kwargs,
        )
