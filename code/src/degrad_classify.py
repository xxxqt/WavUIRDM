import math
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
"""
Functions for building the BottleneckBlock from Detectron2.
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
"""

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_norm(norm, out_channels, num_norm_groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(num_norm_groups, channels),
            "LN": lambda channels: LayerNorm(channels),
            "BN": lambda channels: nn.BatchNorm2d(channels),
        }[norm]
    return norm(out_channels)


class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.
    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="GN",
        stride_in_1x1=False,
        dilation=1,
        num_norm_groups=32,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels, num_norm_groups),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels, num_norm_groups),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class ResNet(nn.Module):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """简化 Detectron2 的 ResNet，实现多路 feature 输出"""
        super().__init__()
        self.stem = stem   # 输入 Stem（一般 7×7 Conv+BN+ReLU+MaxPool）
        self.num_classes = num_classes  # 若不为 None，则额外加 avgpool+FC 做分类
        # 记录各层输出的 stride / channel
        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        # 构建 residual stages
        self.stage_names, self.stages = [], []
        # 若仅需要部分特征，裁掉多余 stage，节省显存
        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [
                    {"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0)
                    for f in out_features
                ]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children)
            )
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.
        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.
        Returns:
            list[CNNBlockBase]: a list of block module.
        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )
        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert (
                        newk not in kwargs
                    ), f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(
                    in_channels=in_channels, out_channels=out_channels, **curr_kwargs
                )
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.
        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.
        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        block_class = BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(
            num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels
        ):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret

class Diff_DC(nn.Module):
    def __init__(
        self,
        feature_dims,
        num_res_blocks=2,
        num_classes=3,
    ):
        super().__init__()
        self.feature_dims = feature_dims   # 保存参数供后续使用
        self.conv_embed = nn.Sequential(
            nn.Conv2d(3, feature_dims[0], 7, 1, 3),
            LayerNorm(feature_dims[0]),
        )

        self.bottleneck_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,   # 堆叠数量
                    in_channels=feature_dim,   # 输入通道
                    bottleneck_channels=int(feature_dim * 2),  # 中间瓶颈通道
                    out_channels=feature_dim,   # 输出通道（保持不变）
                    norm="LN",    # 使用 LayerNorm
                    # num_norm_groups=num_norm_groups,
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            out_feature_dim = (
                self.feature_dims[l + 1] if l < len(feature_dims) - 1 else feature_dim
            )
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature_dim, out_feature_dim, 1, bias=False),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(),
                )
            )
        self.last_stage = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=num_res_blocks,
                in_channels=feature_dims[-1],
                bottleneck_channels=int(feature_dims[-1] * 2),
                out_channels=feature_dims[-1],
                norm="LN",
                # num_norm_groups=num_norm_groups,
            )
        )
        self.mixing_weights = nn.Parameter(
            torch.ones(len(self.bottleneck_layers)), requires_grad=True
        )
        self.fc = nn.Linear(feature_dims[-1], num_classes)

    def forward(self, lq, features):
        lq_feats = self.conv_embed(lq)
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i, feature in enumerate(features):
            lq_feats = self.bottleneck_layers[i](lq_feats + mixing_weights[i] * feature)
            lq_feats = self.downsample_layers[i](lq_feats)
        lq_feats = self.last_stage(lq_feats).mean(dim=[-1, -2])
        out = self.fc(lq_feats)
        return out



class Diff_NoImg_DC(nn.Module):

    def __init__(
        self,
        feature_dims,
        num_res_blocks=2,
        num_classes=3,
        downsample=False,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.downsample = downsample

        self.bottleneck_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=int(feature_dim * 2),
                    out_channels=feature_dim,
                    norm="LN",
                    # num_norm_groups=num_norm_groups,
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)

            out_feature_dim = (
                self.feature_dims[l + 1] if l < len(feature_dims) - 1 else feature_dim
            )
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature_dim, out_feature_dim, 1, bias=False),
                    nn.MaxPool2d(2, 2),
                    nn.ReLU(),
                )
            )

        self.last_stage = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=num_res_blocks,
                in_channels=feature_dims[-1],
                bottleneck_channels=int(feature_dims[-1] * 2),
                out_channels=feature_dims[-1],
                norm="LN",
                # num_norm_groups=num_norm_groups,
            )
        )

        self.mixing_weights = nn.Parameter(
            torch.ones(len(self.bottleneck_layers)), requires_grad=True
        )

        self.fc = nn.Linear(feature_dims[-1], num_classes)

    def feature_map_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {x.ndim}D tensor.")
        B, C, H, W = x.shape
        x_flat = x.flatten(2)
        seq = x_flat.permute(0, 2, 1).contiguous()
        return seq

    def forward(self, lq, features):
        # lq_feats = self.conv_embed(lq)
        if self.downsample:
            for i, f in enumerate(features):
                f = self.feature_map_to_sequence(f)
                b, n, c = f.shape
                features[i] = (
                    f.transpose(-1, -2)
                    .contiguous()
                    .view(b, c, int(math.sqrt(n)), int(math.sqrt(n)))
                )
        lq_feats = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i, feature in enumerate(features):
            if i > 0 and self.downsample:
                feature = F.interpolate(feature, scale_factor=1 / (2**i))
            lq_feats = self.bottleneck_layers[i](lq_feats + mixing_weights[i] * feature)
            lq_feats = self.downsample_layers[i](lq_feats)
        lq_feats = self.last_stage(lq_feats).mean(dim=[-1, -2])
        out = self.fc(lq_feats)
        return out
