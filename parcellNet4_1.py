# ParcellNet version 4.1 
# Try to get a 1. global attention module, 2. landmark-rule based module, 3. local attention based module
# This is a very crude implementation currently, only using crude integration
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas
import math

#     self.layer1 = self._make_layer(block, 16, layers[0], down_size=True)
#     self.layer2 = self._make_layer(block, 64, layers[1], stride=2, down_size=True)
#     self.layer3 = self._make_layer(block, 128, layers[2], stride=2, down_size=True)

#     self.att_layer4 = self._make_layer(MulScaleBlock, 512, layers[3], stride=1, down_size=False)
#     self.bn_att = nn.BatchNorm2d(512 * MulScaleBlock.expansion)
#     self.att_conv = nn.Conv2d(512 * MulScaleBlock.expansion, num_classes, kernel_size=1, padding=0,
#                            bias=False)
#     self.bn_att2 = nn.BatchNorm2d(num_classes)

#     self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1,
#                            bias=False)
#     self.att_gap = nn.MaxPool2d(14)
#     self.sigmoid = nn.Sigmoid()
#     self.mtl = make_mtl_block(block_percept, layers, num_classes)
#     self.depth_conv = nn.Conv2d(
#         in_channels=num_classes,
#         out_channels=num_classes,
#         kernel_size=1,
#         stride=1,
#         padding=0,
#         groups=num_classes
#     )

#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)


# def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#         downsample = nn.Sequential(
#             nn.Conv2d(self.inplanes, planes * block.expansion,
#                       kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(planes * block.expansion),
#         )

#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))

#     if down_size:
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
#     else:
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))

#         return nn.Sequential(*layers)


# def forward(self, x):
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)

#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)

#     ax = self.att_layer4(x)
#     ax0 = self.relu(self.att_conv(self.bn_att(ax)))
#     ax = self.bn_att2(self.depth_conv(ax0))
#     self.att = self.sigmoid(ax)
#     bs, cs, ys, xs = ax.shape

#     ax = self.att_conv2(ax)
#     ax = self.att_gap(ax0)
#     ax = self.sigmoid(ax)
#     ax = ax.view(ax.size(0), -1)

#     rx, attention = self.mtl(x, self.att)

#     return ax, rx, attention


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


def generate_map_batch(map, crop_size, map_size, spatial_ratio, fill_coeff, center1_x, center1_y, center2_x, center2_y):
    
    spatial_scale = float(map_size)/crop_size
    half_AU_size = round((map_size - 1) / 2.0 * spatial_ratio)
    centers = [[center1_x, center1_y],
                       [center2_x, center2_y]]
    for center_ind in range(len(centers)):
        AU_center_x = torch.round(centers[center_ind][0] * spatial_scale)
        AU_center_y = torch.round(centers[center_ind][1] * spatial_scale)
        start_w = torch.round(AU_center_x - half_AU_size)
        start_h = torch.round(AU_center_y - half_AU_size)
        end_w = torch.round(AU_center_x + half_AU_size)
        end_h = torch.round(AU_center_y + half_AU_size)        

        # treat landmark coordinates as starting from 0 rather than 1
        zeros_v = torch.zeros(start_h.shape)
        mapsize_v = torch.ones(start_h.shape)*(map_size-1)
        start_h = torch.max(torch.stack((zeros_v, start_h)),0).values
        start_h = torch.min(torch.stack((mapsize_v, start_h)),0).values
        start_w = torch.max(torch.stack((zeros_v, start_w)),0).values
        start_w = torch.min(torch.stack((mapsize_v, start_w)),0).values 
        end_h = torch.max(torch.stack((zeros_v, end_h)),0).values
        end_h = torch.min(torch.stack((mapsize_v, end_h)),0).values 
        end_w = torch.max(torch.stack((zeros_v, end_w)),0).values
        end_w = torch.min(torch.stack((mapsize_v, end_w)),0).values 

        for i in range(map.shape[0]):
            
            h_elements = int(end_h[i])+1 - int(start_h[i])
            w_elements = int(end_w[i])+1 - int(start_w[i])

            subtracted_h = torch.Tensor(range(int(start_h[i]), int(end_h[i])+1)).repeat_interleave(w_elements) - AU_center_y[i]
            subtracted_w = torch.Tensor(range(int(start_w[i]), int(end_w[i])+1)).repeat(1, h_elements) - AU_center_x[i]

            max_vals0 = 1 - (abs(subtracted_h) + abs(subtracted_w)) * fill_coeff / (map_size*spatial_ratio)
            max_vals = max_vals0.squeeze().reshape(h_elements, w_elements)
            max_vals_compare = torch.maximum(max_vals, map[i, int(start_h[i]):int(end_h[i])+1, int(start_w[i]):int(end_w[i])+1])
            map[i, int(start_h[i]):int(end_h[i])+1, int(start_w[i]):int(end_w[i])+1] = max_vals_compare


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock_CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_CBAM, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CBAM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def feature_extractor(in_chs, unit_dim=8):
    """
    Simple feature extractor module
    """
    feat_extractor = nn.Sequential(
        nn.Conv2d(in_chs, unit_dim * 12, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 12),
        nn.ReLU(inplace=True),
        nn.Conv2d(unit_dim * 12, unit_dim * 12, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 12),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(unit_dim * 12, unit_dim * 16, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 16),
        nn.ReLU(inplace=True),
        nn.Conv2d(unit_dim * 16, unit_dim * 16, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(unit_dim * 16, unit_dim * 20, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 20),
        nn.ReLU(inplace=True),
        nn.Conv2d(unit_dim * 20, unit_dim * 20, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(unit_dim * 20),
        nn.ReLU(inplace=True),
    )
    return feat_extractor


class make_mtl_block(nn.Module):

    def __init__(self, block, in_planes, layers, num_tasks, num_sub_branch=1):
        self.num_tasks = num_tasks
        super(make_mtl_block, self).__init__()
        
        self.layer4_1 = self._make_layer(block, 160*block.expansion, layers[3], stride=2, down_size=True)
        self.layer4_2 = self._make_layer(block, 160*block.expansion * block.expansion, layers[3], stride=2, down_size=True)

        self.avgpool = nn.AdaptiveMaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

        output = [nn.Linear(640 * block.expansion * num_sub_branch, 1) for _ in range(self.num_tasks)]
        self.output = nn.ModuleList(output)

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        self.inplanes = 160

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x, att_elem_g):
        pred = []
        attention = []
        for i in range(self.num_tasks):
            bs, cs, ys, xs = att_elem_g.shape
            item_att = att_elem_g[:, i].view(bs, 1, ys, xs)

            attention.append(item_att)

            sh = item_att * x
            sh += x

            sh = self.layer4_1(sh)
            sh = self.layer4_2(sh)
            sh = self.avgpool(sh)
            sh = sh.view(sh.size(0), -1)

            sh = self.output[i](sh)
            sh = nn.Sigmoid()(sh)
            pred.append(sh)

        pred = torch.stack(pred,1)
        attention = torch.concat(attention, 1)
        return pred, attention


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


class ResNet(nn.Module):

    def __init__(self, in_chs, land_num, block=BasicBlock, block_percept=BasicBlock, layers=[2,2,2,2], unit_dim=8, num_classes=40, map_size=19, crop_size=224):
        super(ResNet, self).__init__()

        self.map_size = map_size
        self.num_classes = num_classes
        self.land_num = land_num
        self.crop_size = crop_size
        self.spatial_ratio = 0.14
        self.fill_coeff = 0.56

        self.feat_extract = feature_extractor(in_chs=in_chs, unit_dim=8)

        # Landmark Extraction Module
        self.land_pool = nn.AdaptiveMaxPool2d((7, 7))
        self.land_FC = nn.Sequential(
            nn.Linear(7840, unit_dim * 64),
            nn.Linear(unit_dim * 64, land_num * 2)
        )
        self.land_refine = LocalAttentionRefine(au_num=num_classes, unit_dim=8)

        # Global AU extraction Module    
        self.inplanes = unit_dim*20

        self.g_layer1 = self._make_layer(block, unit_dim*20, layers[0], down_size=True)
        self.g_layer2 = self._make_layer(block, unit_dim*20*block.expansion, layers[1], down_size=True)
        self.bn_att = nn.BatchNorm2d(unit_dim * 20 * block.expansion)
        self.att_conv = nn.Conv2d(unit_dim * 20 * block.expansion, num_classes, kernel_size=1, padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn_att2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1,
                               bias=False)
        self.depth_conv = nn.Conv2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=num_classes
        )
        self.att_gap = nn.AdaptiveMaxPool2d(1)

        self.mtl = make_mtl_block(block_percept, unit_dim*20, layers, num_classes)

        # self.g_layer3 = self._make_layer(block, 256, layers[2], stride=2, down_size=True)

        # self.bn_att = nn.BatchNorm2d(256 * block.expansion)
        # self.att_conv = nn.Conv2d(256 * block.expansion, num_classes, kernel_size=1, padding=0,
        #                        bias=False)
        # self.bn_att2 = nn.BatchNorm2d(num_classes)
        # self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1,
        #                        bias=False)
        # self.relu = nn.ReLU(inplace=True)
        # self.depth_conv = nn.Conv2d(
        #     in_channels=num_classes,
        #     out_channels=num_classes,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     groups=num_classes
        # )
        # self.att_gap = nn.MaxPool2d(14)

    def forward(self, x):

        x = self.feat_extract(x) # global feature, 54 x 54
        x_land = self.land_pool(x)
        x_land = self.land_FC(x_land.reshape(x.shape[0], -1))
        
        aus_map = torch.zeros((x.size(0), self.num_classes, self.map_size+8, self.map_size+8))
        
        for i in range(x.size(0)):
            land_array = x_land[i,:]
            land_array = land_array.data.cpu().numpy()
            str_dt = np.append(land_array[0:len(land_array):2], land_array[1:len(land_array):2])
            arr2d = np.array(str_dt).reshape((2, self.land_num))
            ruler = abs(arr2d[0, 22] - arr2d[0, 25])

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
            if self.num_classes == 12:
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

        aus_map = aus_map.to(x.device).detach()
                
        aus_mapped_refine = self.land_refine(aus_map)

        # Global branch
        x_att = self.g_layer1(x)
        x_att = self.g_layer2(x_att)

        # Fuse here, fuse with ax0
        ax0 = self.relu(self.att_conv(self.bn_att(x_att)))
        ax0 = aus_mapped_refine + ax0 * aus_mapped_refine
        # ax0 *= aus_mapped_refine.detach()

        # Attention
        ax = self.bn_att2(self.depth_conv(ax0))
        att = nn.Sigmoid()(ax) # This is the attention module. check the size

        # Prediction
        ax = self.att_conv2(ax0)
        ax = self.att_gap(ax)
        ax = nn.Sigmoid()(ax)
        ax = ax.view(ax.size(0), -1)

        rx, attention = self.mtl(x, att)

        return  x_land, ax, rx, attention

    def _make_layer(self, block, planes, blocks, stride=1, down_size=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        if down_size:
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)


if __name__ == '__main__':
    myDat = torch.rand((10, 3, 176, 176))
    myNet = ResNet(in_chs=3, land_num=68, block=BasicBlock, block_percept=BasicBlock, layers=[2,2,2,2], unit_dim=8, num_classes=40, map_size=44)
    out1, out2, out3, out4 = myNet(myDat)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
