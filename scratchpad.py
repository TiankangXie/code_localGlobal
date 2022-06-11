import torch
import torch.nn as nn
import numpy as np


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
        nn.MaxPool2d(kernel_size=3, stride=1),
    )
    return feat_extractor


class toyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feat_extract = feature_extractor(in_chs=3, unit_dim=8)
    def forward(self, x):
        return self.feat_extract(x)


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


def generate_map_v2(map, crop_size, map_size, spatial_ratio, fill_coeff, center1_x, center1_y, center2_x, center2_y):
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

        # for i in range(map.shape[0]):
            
        h_elements = int(end_h)+1 - int(start_h)
        w_elements = int(end_w)+1 - int(start_w)

        subtracted_h = torch.Tensor(range(int(start_h), int(end_h)+1)).repeat_interleave(w_elements).numpy() - AU_center_y
        subtracted_w = torch.Tensor(range(int(start_w), int(end_w)+1)).repeat(1, h_elements).numpy() - AU_center_x

        max_vals0 = 1 - (abs(subtracted_h) + abs(subtracted_w)) * fill_coeff / (map_size*spatial_ratio)
        max_vals = max_vals0.squeeze().reshape(h_elements, w_elements)
        max_vals_compare = np.maximum(max_vals, map[int(start_h):int(end_h)+1, int(start_w):int(end_w)+1])
        map[int(start_h):int(end_h)+1, int(start_w):int(end_w)+1] = max_vals_compare
        

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


if __name__ == '__main__':

    import time
    map1 = np.zeros((44, 44))
    map2 = np.zeros((44, 44))
    crop_size=176
    map_size=44
    spatial_ratio=0.25
    fill_coeff=0.14
    center1_x = 4
    center1_y = 6
    center2_x = 12
    center2_y = 20

    generate_map_v2(map=map1, crop_size=crop_size, map_size=map_size, spatial_ratio=spatial_ratio, \
                    fill_coeff=fill_coeff, center1_x=center1_x, center1_y=center1_y, \
                    center2_x=center2_x, center2_y=center2_y)

    generate_map(map=map2, crop_size=crop_size, map_size=map_size, spatial_ratio=spatial_ratio, \
                    fill_coeff=fill_coeff, center1_x=center1_x, center1_y=center1_y, \
                    center2_x=center2_x, center2_y=center2_y)

    print(np.allclose(map1, map2))
    print('help')    
    
    map3 = torch.zeros((4, 44, 44))
    center1_x=torch.Tensor([5,6,7,8])
    center1_y=torch.Tensor([5,6,7,8])
    center2_x=torch.Tensor([34,32,30,29])
    center2_y=torch.Tensor([34,32,30,29])

    start = time.time()
    generate_map_batch(map=map3, crop_size=crop_size, map_size=map_size, \
                    spatial_ratio=spatial_ratio,fill_coeff=fill_coeff,center1_x=center1_x,\
                    center1_y=center1_y,center2_x=center2_x,center2_y=center2_y)
    end = time.time()
    print(end - start)

    map4 = torch.zeros((4, 44, 44))
    center1_x=torch.Tensor([5,6,7,8])
    center1_y=torch.Tensor([5,6,7,8])
    center2_x=torch.Tensor([34,32,30,29])
    center2_y=torch.Tensor([34,32,30,29])

    start = time.time()
    for i in range(4):
        generate_map(map=map4[i], crop_size=crop_size, map_size=map_size, \
                        spatial_ratio=spatial_ratio,fill_coeff=fill_coeff,center1_x=center1_x[i],\
                        center1_y=center1_y[i],center2_x=center2_x[i],center2_y=center2_y[i])
    end = time.time()
    print(end - start)

    print(torch.allclose(map3, map4))