import math
import numpy as np
import scikitplot as skplt
import os
from typing import Optional
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(
            f"labels must be of the same dtype torch.int64. Got: {labels.dtype}"
        )

    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """Performs affine transformation to align the images by eyes.
    Performs affine alignment including eyes.

    Args:
        img: gray or RGB
        img_land: 68 system flattened landmarks, shape:(136)
        box_enlarge: relative size of face on the image. Smaller value indicate larger proportion
        img_size = output image size
    Return:
        aligned_img: the aligned image
        new_land: the new landmarks
    """
    leftEye0 = (
        img_land[2 * 36]
        + img_land[2 * 37]
        + img_land[2 * 38]
        + img_land[2 * 39]
        + img_land[2 * 40]
        + img_land[2 * 41]
    ) / 6.0
    leftEye1 = (
        img_land[2 * 36 + 1]
        + img_land[2 * 37 + 1]
        + img_land[2 * 38 + 1]
        + img_land[2 * 39 + 1]
        + img_land[2 * 40 + 1]
        + img_land[2 * 41 + 1]
    ) / 6.0
    rightEye0 = (
        img_land[2 * 42]
        + img_land[2 * 43]
        + img_land[2 * 44]
        + img_land[2 * 45]
        + img_land[2 * 46]
        + img_land[2 * 47]
    ) / 6.0
    rightEye1 = (
        img_land[2 * 42 + 1]
        + img_land[2 * 43 + 1]
        + img_land[2 * 44 + 1]
        + img_land[2 * 45 + 1]
        + img_land[2 * 46 + 1]
        + img_land[2 * 47 + 1]
    ) / 6.0
    deltaX = rightEye0 - leftEye0
    deltaY = rightEye1 - leftEye1
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat(
        [
            [leftEye0, leftEye1, 1],
            [rightEye0, rightEye1, 1],
            [img_land[2 * 30], img_land[2 * 30 + 1], 1],
            [img_land[2 * 48], img_land[2 * 48 + 1], 1],
            [img_land[2 * 54], img_land[2 * 54 + 1], 1],
        ]
    )
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(
        max(mat2[:, 1]) - min(mat2[:, 1])
    ):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat(
        [
            [scale, 0, scale * (halfSize - cx)],
            [0, scale, scale * (halfSize - cy)],
            [0, 0, 1],
        ]
    )
    mat = mat3 * mat1
    aligned_img = cv2.warpAffine(
        img,
        mat[0:2, :],
        (img_size, img_size),
        cv2.INTER_LINEAR,
        borderValue=(128, 128, 128),
    )
    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:, 0], new_land[:, 1]))).astype(int)

    return aligned_img, new_land


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class multiclassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean", device=None) -> None:
        super(multiclassFocalLoss, self).__init__()
        self.gamma = gamma.to(device)
        self.alpha = alpha.to(device)
        self.reduce = reduction
        self.criterion = nn.BCEWithLogitsLoss(weight=alpha, reduction="none")
        self.device = device

    def forward(self, inputs, targets):

        mask_1 = torch.ones((targets.shape)).to(self.device)
        miss_idc = torch.where(targets == 9)
        mask_1[miss_idc] = 0
        miss_idc2 = torch.where(targets == 999)
        mask_1[miss_idc2] = 0
        miss_idc3 = torch.where(targets == -1)
        mask_1[miss_idc3] = 0

        ce_loss = self.criterion(inputs, targets.float())
        ce_loss *= mask_1  # mask out index
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduce == "mean":
            return focal_loss.mean()
        elif self.reduce == "sum":
            return focal_loss.sum()
        elif self.reduce == "none":
            return focal_loss
        else:
            raise ValueError("not valid reduction method")


class dice_loss(nn.Module):
    def __init__(self, device, smooth=1):
        super(dice_loss, self).__init__()
        self.device = device
        self.smooth = smooth

    def __call__(self, outs, labels):
        mask_1 = torch.ones((labels.shape)).to(self.device)
        miss_idc = torch.where(labels == 9)
        mask_1[miss_idc] = 0
        miss_idc2 = torch.where(labels == 999)
        mask_1[miss_idc2] = 0
        miss_idc3 = torch.where(labels == -1)
        mask_1[miss_idc3] = 0

        mask_flat = mask_1.contiguous().view(-1)
        iflat = outs.contiguous().view(-1)[np.where(mask_flat == 1)]
        tflat = labels.contiguous().view(-1)[np.where(mask_flat == 1)]
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        loss_dice = 1 - (
            (2.0 * intersection + self.smooth) / (A_sum + B_sum + self.smooth)
        ) / iflat.size(0)
        return loss_dice


class CenterLoss(nn.Module):
    """Center loss.
    https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    Use Case:
        MNIST: center_loss = CenterLoss(num_classes=10, feat_dim=2, use_gpu=True)
    """

    def __init__(self, num_classes=10, feat_dim=2, device="cpu"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # if self.use_gpu:
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim).to(device)
        )
        # else:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            labels -> [0,1] -> not one-hot encoded
        """
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)

        # if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size

        return loss


def au_softmax_loss(
    input, target, device="cuda", weight=None, size_average=True, reduce=True
):
    """
    https://github.com/ZhiwenShao/PyTorch-JAANet/blob/c639c66c3086c0c6d1999229f4ce71aef5e2821e/util.py

    input: of shape (B, 2, AU)
    target: of shape (B)
    target:

    """

    # Step 1 mask out the missing
    mask_1 = torch.ones((target.shape)).to(device)
    miss_idc = torch.where(target == 9)
    mask_1[miss_idc] = 0
    miss_idc2 = torch.where(target == 999)
    mask_1[miss_idc2] = 0
    miss_idc3 = torch.where(target == -1)
    mask_1[miss_idc3] = 0
    mask_2 = (1 - mask_1).to(device, dtype=torch.bool)
    target.masked_fill_(mask_2, 0)

    # One-hot encode
    # target = F.one_hot(target, 2)

    # Input should be LogSoftmax
    m = nn.LogSoftmax(dim=1)
    classify_loss = nn.NLLLoss(size_average=size_average, reduce=reduce)

    for i in range(input.size(2)):
        t_input = input[:, :, i]
        t_target = target[:, i]
        t_loss = classify_loss(m(t_input), t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]

        if size_average:
            t_loss = torch.mean(t_loss * mask_1[:, i])
        else:
            t_loss = torch.sum(t_loss * mask_1[:, i])

        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def min_max(x, axis=None):
    min = 0  # x.min(axis=axis, keepdims=True)
    max = 1  # x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)


import numpy as np
from torchvision import transforms
from PIL import Image


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class land_transform(object):
    def __init__(self, img_size, flip_reflect):
        self.img_size = img_size
        self.flip_reflect = flip_reflect.astype(int) - 1

    def __call__(self, land, flip, offset_x, offset_y):
        land[0:len(land):2] = land[0:len(land):2] - offset_x
        land[1:len(land):2] = land[1:len(land):2] - offset_y
        # change the landmark orders when flipping
        if flip:
            land[0:len(land):2] = self.img_size - 1 - land[0:len(land):2]
            land[0:len(land):2] = land[0:len(land):2][self.flip_reflect]
            land[1:len(land):2] = land[1:len(land):2][self.flip_reflect]

        return land


class image_train(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def image_test(crop_size=176):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
    

if __name__ == "__main__":

    # outs = torch.rand((10, 2, 14)).to('cuda')
    # preds = torch.from_numpy(np.array([1,0,1,1,1,0,0,1,0,1,1,9,9,1])).repeat(10,1).to('cuda')
    # au_softmax_loss(input=outs, target=preds)

    outs = torch.rand((14, 128))  # .to('cuda')
    preds = torch.from_numpy(
        np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    )  # .to('cuda')#.repeat(10,1).to('cuda')
    center_loss = CenterLoss(num_classes=2, feat_dim=128, device='cpu')
    los1 = center_loss(outs, preds)
    print(los1)
    # loss1 = dice_loss(device='cpu')
    # loss1(outs, preds)
