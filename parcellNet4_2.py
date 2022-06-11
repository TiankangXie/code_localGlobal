## The second edition of ParcellNet
# 1. Global Feature
# 2. Landmark Feature
# 3. Fuse the global features with the landmark features.
# 4. global attention
# 5. local AU specific attention 
# 6. Frame it as a multitask learning problem

# Maybe instead of ResNet, try VGG style feature extractor?
# This time try get a light-weight model, try to see what is the minimum 
# number of architecture required to perform good predictions
from threading import local
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import math
import time


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
        mapsize_v = (torch.ones(start_h.shape)*(map_size-1))
        
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
            subtracted_w = torch.Tensor(range(int(start_w[i]), int(end_w[i])+1)).repeat(1, h_elements).squeeze() - AU_center_x[i]

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


# Start with easy feature extractor
# VGG style network
class feature_Extract(nn.Module):

    def __init__(self, in_chs) -> None:
        super().__init__()
        self.conv_1_1 = nn.Conv2d(in_chs, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        return x


class landmark_Module(nn.Module):
    def __init__(self, in_chs=64, num_land=49) -> None:
        super().__init__()
        self.extraction = nn.Sequential(
            nn.Conv2d(in_chs, 128, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 256, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*7*7, 1024),
            nn.Linear(1024, num_land*2)
        )
    def forward(self, x):
        x = self.extraction(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


class LocalAttentionRefine(nn.Module):
    def __init__(self, au_num, latent_dim=8):
        super(LocalAttentionRefine, self).__init__()

        self.local_aus_attention = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(1, latent_dim, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, bias=True),
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


class localResNet(nn.Module):
    def __init__(self, block_g, in_chs, num_land, num_aus, map_size, crop_size=176, n_layers=[2,2], device='cuda') -> None:
        super().__init__()
        
        self.num_classes = num_aus
        self.map_size = map_size
        self.land_num = num_land
        self.crop_size = crop_size
        self.spatial_ratio = 0.14
        self.fill_coeff = 0.56

        self.inplanes = 64
        self.feature_extract = feature_Extract(in_chs=in_chs) #256
        self.land_classifier = landmark_Module(in_chs=64, num_land=num_land)
        self.attention_refine = LocalAttentionRefine(au_num=num_aus, latent_dim=64)

        self.l_layer1 = self._make_layer(block=block_g, planes=256, blocks=n_layers[0], stride=2)
        self.l_layer2 = self._make_layer(block=block_g, planes=256, blocks=n_layers[1])

        self.bn_att = nn.BatchNorm2d(256 * block_g.expansion)
        self.att_conv = nn.Conv2d(256 * block_g.expansion, num_aus, kernel_size=1, padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn_att2 = nn.BatchNorm2d(num_aus)
        self.att_conv2 = nn.Conv2d(num_aus, num_aus, kernel_size=3, padding=1,
                               bias=False)
        self.max_pool = nn.MaxPool2d((3,3))
        self.final_classifier = nn.Sequential(
            nn.Linear(num_aus*14*14, 512),
            nn.Linear(512, num_aus*2)
        )
        self.m = nn.LogSoftmax(dim=1)
        self.device = device

    def forward_old(self, x):
        x = self.feature_extract(x)
        x_land = self.land_classifier(x)
        land_len = x_land.shape[1]

        aus_map = torch.zeros((x.size(0), self.num_classes, self.map_size+8, self.map_size+8))
        # Need to modify this part to speed up GPU time 
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
        aus_mapped_refine = self.attention_refine(aus_map)
        
        x = self.l_layer1(x)
        x = self.l_layer2(x)

        # Prediction
        ax = self.relu(self.att_conv(self.bn_att(x)))
        ax = ax + ax * aus_mapped_refine
        # Prediction
        ax = self.att_conv2(ax)
        ax = self.max_pool(ax)
        ax = ax.reshape(x.shape[0], -1)
        ax = self.final_classifier(ax)

        return x_land, aus_mapped_refine, aus_map, self.m(ax.reshape(x.shape[0], 2, -1));

    def forward(self, x):

        x = self.feature_extract(x)
        x_land = self.land_classifier(x)

        aus_map = torch.zeros((x.size(0), self.num_classes, self.map_size+8, self.map_size+8))
    
        land_array = x_land
        land_array = land_array.data.cpu()
        str_dt = torch.cat([land_array[:, 0:x_land.shape[-1]:2], land_array[:, 1:x_land.shape[-1]:2]], dim=1)
        arr2d = str_dt.reshape((x.shape[0], 2, self.land_num))
        ruler = abs(arr2d[:, 0, 22] - arr2d[:, 0, 25])

        generate_map_batch(aus_map[:,0,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                            arr2d[:,0,4], arr2d[:,1,4]-ruler/2, arr2d[:,0,5], arr2d[:,1,5]-ruler/2)

        # au2
        generate_map_batch(aus_map[:,1,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                            arr2d[:,0,1],arr2d[:,1,1]-ruler/3,arr2d[:,0,8],arr2d[:,1,8]-ruler/3)

        # au4
        generate_map_batch(aus_map[:,2,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                            arr2d[:,0,2],arr2d[:,1,2]+ruler/3,arr2d[:,0,7],arr2d[:,1,7]+ruler/3)

        # au6
        generate_map_batch(aus_map[:,3,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                            arr2d[:,0,24],arr2d[:,1,24]+ruler,arr2d[:,0,29],arr2d[:,1,29]+ruler)
        # for bp4d
        if self.num_classes == 12:
            # au7
            generate_map_batch(aus_map[:,4,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                arr2d[:,0,21],arr2d[:,1,21],arr2d[:,0,26],arr2d[:,1,26])

            # au10
            generate_map_batch(aus_map[:,5,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                arr2d[:,0,43],arr2d[:,1,43],arr2d[:,0,45],arr2d[:,1,45])

            # au12 au14 au15
            generate_map_batch(aus_map[:,6,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                arr2d[:,0,31],arr2d[:,1,31],arr2d[:,0,37],arr2d[:,1,37])
            aus_map[:,7,...] =aus_map[:,6,...]
            aus_map[:,8,...] =aus_map[:,6,...]

            # au17
            generate_map_batch(aus_map[:,9,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                arr2d[:,0,39],arr2d[:,1,39]+ruler/2,arr2d[:,0,41],arr2d[:,1,41]+ruler/2)

            # au23 au24
            generate_map_batch(aus_map[:,10,...], self.crop_size, self.map_size+8, self.spatial_ratio, self.fill_coeff,
                                arr2d[:,0,34],arr2d[:,1,34],arr2d[:,0,40],arr2d[:,1,40])
            aus_map[:,11,...] = aus_map[:,10,...]

        aus_map = aus_map.to(x.device).detach()
        aus_mapped_refine = self.attention_refine(aus_map)
        
        x = self.l_layer1(x)
        x = self.l_layer2(x)

        # Prediction
        ax = self.relu(self.att_conv(self.bn_att(x)))
        ax = ax + ax * aus_mapped_refine
        # Prediction
        ax = self.att_conv2(ax)
        ax = self.max_pool(ax)
        ax = ax.reshape(x.shape[0], -1)
        ax = self.final_classifier(ax)

        return x_land, aus_mapped_refine, aus_map, self.m(ax.reshape(x.shape[0], 2, -1));

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


def get_resnet(in_chs, num_lands, num_classes, map_size=44, n_layers=[2,2], crop_size=176):
    
    net = localResNet(block_g=BasicBlock_CBAM, in_chs=in_chs, num_land=num_lands, \
                    num_aus=num_classes, map_size=map_size, n_layers=n_layers, crop_size=crop_size)
    
    return net


if __name__ == '__main__':
    myDat = torch.rand((16, 3, 176, 176)).to('cuda:0')
    myNet3 = localResNet(block_g=BasicBlock_CBAM, in_chs=3, num_land=49, num_aus=12, map_size=44, n_layers=[2,2]).to('cuda:0')
    start = time.time()
    out3_1 = myNet3.forward(myDat)
    end = time.time()
    print(end - start)

    start = time.time()
    out3_2 = myNet3.forward_old(myDat)
    end = time.time()
    print(end - start)
    # print(out3_1[2].shape)
    # print(out3_2[2].shape)
    print(torch.allclose(out3_1[2], out3_2[2]))
    print(torch.mean(out3_1[2]-out3_2[2]))
