## parcellation based net. Every parcell will have a smaller
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional

# from FER_GAN.code.faceParsing.parsing_NN.experimental2 import Bottleneck


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype="normal",
    ):
        """Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)

        return x_out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttentionBlock(nn.Module):
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MulScaleBlock(nn.Module):
    __constants__ = ["downsample"]

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)

        self.conv2_2_1 = conv3x3(scale_width, scale_width)
        self.bn2_2_1 = norm_layer(scale_width)
        self.conv2_2_2 = conv3x3(scale_width, scale_width)
        self.bn2_2_2 = norm_layer(scale_width)
        self.conv2_2_3 = conv3x3(scale_width, scale_width)
        self.bn2_2_3 = norm_layer(scale_width)
        self.conv2_2_4 = conv3x3(scale_width, scale_width)
        self.bn2_2_4 = norm_layer(scale_width)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        ##########################################################
        out_1_1 = self.conv1_2_1(sp_x[0])
        out_1_1 = self.bn1_2_1(out_1_1)
        out_1_1_relu = self.relu(out_1_1)
        out_1_2 = self.conv1_2_2(out_1_1_relu + sp_x[1])
        out_1_2 = self.bn1_2_2(out_1_2)
        out_1_2_relu = self.relu(out_1_2)
        out_1_3 = self.conv1_2_3(out_1_2_relu + sp_x[2])
        out_1_3 = self.bn1_2_3(out_1_3)
        out_1_3_relu = self.relu(out_1_3)
        out_1_4 = self.conv1_2_4(out_1_3_relu + sp_x[3])
        out_1_4 = self.bn1_2_4(out_1_4)
        output_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)

        out_2_1 = self.conv2_2_1(sp_x[0])
        out_2_1 = self.bn2_2_1(out_2_1)
        out_2_1_relu = self.relu(out_2_1)
        out_2_2 = self.conv2_2_2(out_2_1_relu + sp_x[1])
        out_2_2 = self.bn2_2_2(out_2_2)
        out_2_2_relu = self.relu(out_2_2)
        out_2_3 = self.conv2_2_3(out_2_2_relu + sp_x[2])
        out_2_3 = self.bn2_2_3(out_2_3)
        out_2_3_relu = self.relu(out_2_3)
        out_2_4 = self.conv2_2_4(out_2_3_relu + sp_x[3])
        out_2_4 = self.bn2_2_4(out_2_4)
        output_2 = torch.cat([out_2_1, out_2_2, out_2_3, out_2_4], dim=1)

        out = output_1 + output_2

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class localGlobalNet(nn.Module):
    def __init__(
        self, in_chs, num_classes, layers, num_cuts=4, channel_multiplier=1
    ) -> None:
        ini_inplane = 16
        super(localGlobalNet, self).__init__()
        # =======FEATURE EXTRACTION LAYER===============
        self.num_cuts = num_cuts
        self.conv1 = nn.Conv2d(
            in_chs, ini_inplane, kernel_size=15, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ini_inplane)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, ini_inplane, 32, layers[0])
        # self.layer2 = self._make_layer(BasicBlock, 32, 64, layers[1], stride=2)
        # =======DETAIL EXTRACTION LAYER=================
        self.layer3_1s = nn.ModuleList(
            [
                self._make_layer(BasicBlock, 32, 64, layers[2], stride=1)
                # self._make_layer(AttentionBlock, 128, 256, layers[2], stride=2)
                for _ in range(num_cuts * num_cuts)
            ]
        )
        # self.layer4_1s = nn.ModuleList(
        #     [
        #         self._make_layer(BasicBlock, 64, 128, layers[3], stride=1)
        #         # self._make_layer(AttentionBlock, 256, 512, layers[3], stride=2)
        #         for _ in range(num_cuts * num_cuts)
        #     ]
        # )

        # In this branch, each BasicBlock replaced by MulScaleBlock.
        self.layer3_2 = self._make_layer(
            BasicBlock, 32, 64, layers[2], stride=1
        )  # , baseWidth=16)
        # self.layer4_2 = self._make_layer(
        #     BasicBlock, 64, 128, layers[3], stride=1
        # )  # , baseWidth=16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_multiplier = channel_multiplier
        if channel_multiplier == 1:
            self.fc1 = nn.Linear(64, num_classes)
            self.fc2 = nn.Linear(64, num_classes)
        else:
            self.fc1 = nn.Linear(64, num_classes * channel_multiplier)
            self.fc2 = nn.Linear(64, num_classes * channel_multiplier)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, **kwargs):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), norm_layer(planes)
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, **kwargs))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, **kwargs))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, inplanes, planes, blocks, stride=1, **kwargs):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, **kwargs))
        return nn.Sequential(*layers)

    # def _make_layer2(self, block, inplanes, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(inplanes, planes, stride, downsample=downsample,
    #                     stype='stage', baseWidth = 26, scale=4))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, baseWidth = 26, scale=4))

    #     return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        # split and pass======================================
        patch_size = x.shape[-1] // self.num_cuts
        reorganized_tensor_full = None
        for i in range(self.num_cuts):
            reorganized_tenor_tmp = None
            for j in range(self.num_cuts):
                tmp_out = x[
                    :,
                    :,
                    int(i * patch_size) : int((i + 1) * patch_size),
                    int(j * patch_size) : int((j + 1) * patch_size),
                ]
                tmp_out = self.layer3_1s[int(i * self.num_cuts + j)](tmp_out)
                # tmp_out = self.layer4_1s[int(i * self.num_cuts + j)](tmp_out)
                if reorganized_tenor_tmp is None:
                    reorganized_tenor_tmp = tmp_out
                else:
                    reorganized_tenor_tmp = torch.cat(
                        [reorganized_tenor_tmp, tmp_out], dim=3
                    )
            if reorganized_tensor_full is None:
                reorganized_tensor_full = reorganized_tenor_tmp
            else:
                reorganized_tensor_full = torch.cat(
                    [reorganized_tensor_full, reorganized_tenor_tmp], dim=2
                )
        reorganized_tensor_full = self.avgpool(reorganized_tensor_full)
        reorganized_tensor_full = torch.flatten(reorganized_tensor_full, 1)
        reorganized_tensor_full = self.fc1(reorganized_tensor_full)
        # branch 2 ############################################
        branch_2_out = self.layer3_2(x)
        # branch_2_out = self.layer4_2(branch_2_out)
        branch_2_out = self.avgpool(branch_2_out)
        branch_2_out = torch.flatten(branch_2_out, 1)
        branch_2_out = self.fc2(branch_2_out)

        # branch fusion network

        if self.channel_multiplier > 1:
            return reorganized_tensor_full.reshape(
                x.shape[0], 2, -1
            ), branch_2_out.reshape(x.shape[0], 2, -1)
        else:
            return reorganized_tensor_full, branch_2_out


def generate_map(map, crop_size, map_size, spatial_ratio, fill_coeff, center1_x, center1_y, center2_x, center2_y):
    spatial_scale = float(map_size)/crop_size
    half_AU_size = round((map_size - 1) / 2.0 * spatial_ratio)

    centers= np.array([[center1_x, center1_y],
                       [center2_x, center2_y]])
    for center_ind in range(centers.shape[0]):
        AU_center_x = round(centers[center_ind,0] * spatial_scale)
        AU_center_y = round(centers[center_ind,1] * spatial_scale)
        start_w = round(AU_center_x - half_AU_size)
        start_h = round(AU_center_y - half_AU_size)
        end_w = round(AU_center_x + half_AU_size)
        end_h = round(AU_center_y + half_AU_size)

        # treat landmark coordinates as starting from 0 rather than 1
        start_h = max(start_h, 0)
        start_h = min(start_h, map_size - 1)
        start_w = max(start_w, 0)
        start_w = min(start_w, map_size - 1)
        end_h = max(end_h, 0)
        end_h = min(end_h, map_size - 1)
        end_w = max(end_w, 0)
        end_w = min(end_w, map_size - 1)

        for h in range(int(start_h), int(end_h)+1):
            for w in range(int(start_w), int(end_w)+1):
                map[h,w]=max(1 - (abs(h - AU_center_y) + abs(w - AU_center_x)) *
                                 fill_coeff / (map_size*spatial_ratio), map[h,w])

                                 
class AlignNet(nn.Module):
    def __init__(self, crop_size, map_size, au_num, land_num, input_dim, unit_dim=8,
                 spatial_ratio=0.14, fill_coeff= 0.56):
        super(AlignNet, self).__init__()

        self.align_feat = HLFeatExtractor(input_dim=input_dim, unit_dim=unit_dim)
        self.align_output = nn.Sequential(
            nn.Linear(4000, unit_dim * 64),
            nn.Linear(unit_dim * 64, land_num * 2)
        )
        self.crop_size = crop_size
        self.map_size = map_size
        self.au_num = au_num
        self.land_num = land_num
        self.spatial_ratio = spatial_ratio
        self.fill_coeff = fill_coeff

    def forward(self, x):
        align_feat_out = self.align_feat(x)
        align_feat = align_feat_out.view(align_feat_out.size(0), -1)
        align_output = self.align_output(align_feat)

        aus_map = torch.zeros((align_output.size(0), self.au_num, self.map_size+8, self.map_size+8))
        for i in range(align_output.size(0)):
            land_array = align_output[i,:]
            land_array = land_array.data.cpu().numpy()
            str_dt = np.append(land_array[0:len(land_array):2], land_array[1:len(land_array):2])
            arr2d = np.array(str_dt).reshape((2, self.land_num))
            ruler = abs(arr2d[0, 22] - arr2d[0, 25])

            #au1
            generate_map(aus_map[i,0], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,4], arr2d[1,4]-ruler/2, arr2d[0,5], arr2d[1,5]-ruler/2)


            # au2
            generate_map(aus_map[i,1], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,1],arr2d[1,1]-ruler/3,arr2d[0,8],arr2d[1,8]-ruler/3)

            # au4
            generate_map(aus_map[i,2], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,2],arr2d[1,2]+ruler/3,arr2d[0,7],arr2d[1,7]+ruler/3)

            # au6
            generate_map(aus_map[i,3], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                               arr2d[0,24],arr2d[1,24]+ruler,arr2d[0,29],arr2d[1,29]+ruler)
            # for bp4d
            if self.au_num == 12:
                # au7
                generate_map(aus_map[i,4], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,21],arr2d[1,21],arr2d[0,26],arr2d[1,26])

                # au10
                generate_map(aus_map[i,5], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,43],arr2d[1,43],arr2d[0,45],arr2d[1,45])

                # au12 au14 au15
                generate_map(aus_map[i,6], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,31],arr2d[1,31],arr2d[0,37],arr2d[1,37])
                aus_map[i,7] =aus_map[i,6]
                aus_map[i,8] =aus_map[i,6]

                # au17
                generate_map(aus_map[i,9], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,39],arr2d[1,39]+ruler/2,arr2d[0,41],arr2d[1,41]+ruler/2)

                # au23 au24
                generate_map(aus_map[i,10], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                   arr2d[0,34],arr2d[1,34],arr2d[0,40],arr2d[1,40])
                aus_map[i, 11] = aus_map[i,10]
            # for disfa
            elif self.au_num == 8:
                # au9
                generate_map(aus_map[i, 4], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0,15],arr2d[1,15]-ruler/2,arr2d[0,17],arr2d[1,17]-ruler/2)
                # au12
                generate_map(aus_map[i, 5], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 31], arr2d[1, 31], arr2d[0, 37], arr2d[1, 37])
                # au25
                generate_map(aus_map[i, 6], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 34], arr2d[1, 34], arr2d[0, 40], arr2d[1, 40])
                # au26
                generate_map(aus_map[i, 7], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 39], arr2d[1, 39] + ruler / 2, arr2d[0, 41], arr2d[1, 41] + ruler / 2)
            # for gft
            elif self.au_num == 10:
                # au10
                generate_map(aus_map[i, 4], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 43], arr2d[1, 43], arr2d[0, 45], arr2d[1, 45])
                # au12 au14 au15
                generate_map(aus_map[i, 5], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 31], arr2d[1, 31], arr2d[0, 37], arr2d[1, 37])
                aus_map[i, 6] = aus_map[i, 5]
                aus_map[i, 7] = aus_map[i, 5]
                # au23 au24
                generate_map(aus_map[i, 8], self.crop_size, self.map_size + 8, self.spatial_ratio, self.fill_coeff,
                             arr2d[0, 34], arr2d[1, 34], arr2d[0, 40], arr2d[1, 40])
                aus_map[i, 9] = aus_map[i, 8]

        return align_feat_out, align_output, aus_map


class LocalAttentionRefine(nn.Module):
    def __init__(self, au_num, unit_dim=8):
        super(LocalAttentionRefine, self).__init__()

        self.local_aus_attention = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(1, unit_dim * 8, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(unit_dim * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(unit_dim * 8, unit_dim * 8, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(unit_dim * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(unit_dim * 8, unit_dim * 8, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(unit_dim * 8),
            nn.ReLU(inplace=True),

            nn.Conv2d(unit_dim * 8, 1, kernel_size=3, stride=1, bias=True),
            nn.Sigmoid(),
        ) for i in range(au_num)])

    def forward(self, x):
        for i in range(len(self.local_aus_attention)):
            initial_au_map = x[:,i,:,:]
            initial_au_map = initial_au_map.unsqueeze(1)
            au_map = self.local_aus_attention[i](initial_au_map)
            if i==0:
                aus_map = au_map
            else:
                aus_map = torch.cat((aus_map, au_map), 1)

        return aus_map


if __name__ == "__main__":
    myNet = localGlobalNet(
        in_chs=3, num_classes=13, layers=[2, 2, 2, 2], num_cuts=4, channel_multiplier=1
    )
    myDat = torch.rand((10, 3, 256, 256))
    out = myNet(myDat)
    print(out[0].shape)
    print(out[1].shape)
