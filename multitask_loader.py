# Multitask loader for facial expressions
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d
import random
import numpy as np
import torch
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


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


## Note: to add new algorithm to it
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


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y : mask_y + mask_height, mask_x : mask_x + mask_width] = 1
    return mask


def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0], img.shape[1]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j : j + side, i : i + side, ...]

    img = cv2.resize(img, (height, width))
    return img


# -> Landmark, AU & Emotion Expressions
def ordered_encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = list(dict.fromkeys(values))
        uniques = np.array(uniques, dtype=values.dtype)
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = np.array([table[v] for v in values])
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s" % str(e))
        return uniques, encoded
    else:
        return uniques


def inflate_parsing(parsing_matrix, stride=0.5):
    """
    Convert parsing matrix into tensor with shape (C, W, H)
    """
    vis_parsing_anno = parsing_matrix.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST
    )

    # This could be changeable depending on the task preferences.
    index_record = [1]
    stacked_arr = None
    for chose_index in index_record:
        sub_arr = np.expand_dims(np.where(vis_parsing_anno == chose_index, 1, 0), 0)
        if stacked_arr is None:
            stacked_arr = sub_arr
        else:
            stacked_arr = np.concatenate([sub_arr, stacked_arr], 0)

    return stacked_arr


# Need to add into the function to fill the inner empty cells!
def inflate_and_fill(parsing_matrix, stride=0.5):
    """
    ONLY FOR A SINGLE INDEX (SKIN)
    """
    vis_parsing_anno = parsing_matrix.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST
    )

    # DO NOT CHANGE THIS INDEX.
    index_record = [1]
    index_row = []
    index_col = []
    index_A1 = np.where(vis_parsing_anno == index_record[0])
    stacked_arr = np.zeros(vis_parsing_anno.shape)
    for i in range(len(index_A1[1]) - 1):
        if index_A1[1][i + 1] > index_A1[1][i]:
            if index_A1[1][i + 1] - 1 > index_A1[1][i]:
                index_col += list(range(index_A1[1][i], index_A1[1][i + 1]))
                index_row += [index_A1[0][i]] * (index_A1[1][i + 1] - index_A1[1][i])
            else:
                index_col += [index_A1[1][i]]
                index_row += [index_A1[0][i]]
        else:
            index_col += [index_A1[1][i]]
            index_row += [index_A1[0][i]]

    stacked_arr[index_row, index_col] = 1
    stacked_arr = np.expand_dims(stacked_arr, 0)

    return stacked_arr


class identity_transform(object):
    """
    Extreme face image transformation
    See more: https://arxiv.org/abs/2104.11116
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        img = self._color_transfer(img)
        # img = np.expand_dims(img,0)
        img = self._reshape(img, self.crop_size)
        img = self._blur_and_sharp(img)
        return img

    def _blur_and_sharp(self, img):
        blur = np.random.randint(0, 2)
        img2 = img.copy()
        output = []
        for i in range(len(img2)):
            if blur:
                ksize = np.random.choice([3, 5, 7, 9])
                output.append(cv2.medianBlur(img2[i], ksize))
            else:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                output.append(cv2.filter2D(img2[i], -1, kernel))
        output = np.stack(output)
        return output

    def _color_transfer(self, img):

        transfer_c = np.random.uniform(0.3, 1.6)

        start_channel = np.random.randint(0, 2)
        end_channel = np.random.randint(start_channel + 1, 4)

        img2 = img.copy()

        img2[:, :, start_channel:end_channel] = np.minimum(
            np.maximum(
                img[:, :, start_channel:end_channel] * transfer_c,
                np.zeros(img[:, :, start_channel:end_channel].shape),
            ),
            np.ones(img[:, :, start_channel:end_channel].shape) * 255,
        )
        return img2

    def perspective_transform(self, img, crop_size=224, pers_size=10, enlarge_size=-10):
        h, w, c = img.shape
        dst = np.array(
            [
                [-enlarge_size, -enlarge_size],
                [-enlarge_size + pers_size, w + enlarge_size],
                [h + enlarge_size, -enlarge_size],
                [h + enlarge_size - pers_size, w + enlarge_size],
            ],
            dtype=np.float32,
        )
        src = np.array(
            [
                [-enlarge_size, -enlarge_size],
                [-enlarge_size, w + enlarge_size],
                [h + enlarge_size, -enlarge_size],
                [h + enlarge_size, w + enlarge_size],
            ]
        ).astype(np.float32())
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            img, M, (crop_size, crop_size), borderMode=cv2.BORDER_REPLICATE
        )
        return warped, M

    def _reshape(self, img, crop_size):
        reshape = np.random.randint(0, 2)
        reshape_size = np.random.randint(15, 25)
        extra_padding_size = np.random.randint(0, reshape_size // 2)
        pers_size = np.random.randint(20, 30) * pow(-1, np.random.randint(2))

        enlarge_size = np.random.randint(20, 40) * pow(-1, np.random.randint(2))
        shape = img.shape
        img2 = img.copy()
        output = []
        # for i in range(len(img2)):
        if reshape:
            im = cv2.resize(
                img2, (shape[0] - reshape_size * 2, shape[1] + reshape_size * 2)
            )
            im = cv2.copyMakeBorder(
                im,
                0,
                0,
                reshape_size + extra_padding_size,
                reshape_size + extra_padding_size,
                cv2.BORDER_REFLECT,
            )
            im = im[
                reshape_size
                - extra_padding_size : shape[0]
                + reshape_size
                + extra_padding_size,
                :,
                :,
            ]
            im, _ = self.perspective_transform(
                im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size
            )
            output.append(im)
        else:
            im = cv2.resize(
                img2, (shape[0] + reshape_size * 2, shape[1] - reshape_size * 2)
            )
            im = cv2.copyMakeBorder(
                im,
                reshape_size + extra_padding_size,
                reshape_size + extra_padding_size,
                0,
                0,
                cv2.BORDER_REFLECT,
            )
            im = im[
                :,
                reshape_size
                - extra_padding_size : shape[0]
                + reshape_size
                + extra_padding_size,
                :,
            ]
            im, _ = self.perspective_transform(
                im, crop_size=crop_size, pers_size=pers_size, enlarge_size=enlarge_size
            )
            output.append(im)

        output = np.stack(output)
        return output


class OrderedLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = ordered_encode_python(y)

    def fit_transform(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = ordered_encode_python(y, encode=True)
        return y


class DataBase(Dataset):
    """
    Parent class for all data loaders
    """

    def __init__(
        self,
        master_file,
        train_mode="train",
        AU_detections=[
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU10",
            "AU12",
            "AU14",
            "AU15",
            "AU17",
            "AU20",
            "AU25",
        ],
        transform=None,
        img_size=256,
        mask_files="/Storage/Projects/lafin_img/mask_data/mask/testing_mask_dataset/",
        mask_type=2,
    ):
        """
        Params:
            master_file: a master csv file, for all image paths and label data
            img_size: if you are not happy with image size, you can resize it
            mask_files: file path to external masks
            mask_type: type of the mask applied to the data
            AU_detections: this is what the Action Unit algo is capable of detecting
        """
        super(DataBase, self).__init__()
        random.seed(1)
        self._name = "Base Loader"
        # self.transforms = transform
        self._IMG_EXTENSIONS = [
            ".jpg",
            ".JPG",
            ".jpeg",
            ".JPEG",
            ".png",
            ".PNG",
            ".ppm",
            ".PPM",
            ".bmp",
            ".BMP",
        ]
        # self._master_root = master_root
        self.train_mode = train_mode
        self.master_file = master_file
        self.img_size = img_size
        self._has_AU = False
        self._has_Emo = False
        self.all_aus = AU_detections
        self.mask_files = glob.glob(mask_files + "*.png")
        self.mask_type = mask_type

    @property
    def _get_name(self):
        return self._name

    def peek_master(self):
        return self.master_file.head()

    def get_label_status(self):
        return f"has_AU:{self._has_AU}; has_emo:{self.has_Emo}"

    def __len__(self):
        return len(self.master_file)

    def load_mask(self, img, index):
        "Load the mask"
        imgh, imgw = self.img_size, self.img_size
        mask_type = self.mask_type  # mask_type ->

        # 50% no mask, 25% random block mask, 25% external mask, for landmark predictor training.
        if mask_type == 5:
            mask_type = 0 if np.random.uniform(0, 1) >= 0.5 else 4

        # no mask
        if mask_type == 0:
            return np.zeros((imgh, imgw))

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # center mask
        if mask_type == 2:
            return create_mask(
                imgw, imgh, imgw // 2, imgh // 2, x=imgw // 4, y=imgh // 4
            )

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_files) - 1)
            mask = cv2.imread(self.mask_files[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8)  # * 255
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = cv2.imread(
                self.mask_files[index % len(self.mask_files)], cv2.IMREAD_GRAYSCALE
            )
            mask = resize(mask, imgh, imgw, centerCrop=False)
            mask = (mask > 0).astype(np.uint8)  # * 255
            return mask

        # Load fixed "random" mask
        if mask_type == 7:
            # mask_index = 23
            # mask = cv2.imread(self.mask_files[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(
                "/Storage/Projects/lafin_img/mask_data/mask/testing_mask_dataset/10651.png",
                cv2.IMREAD_GRAYSCALE,
            )
            mask = resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8)  # * 255
            return mask
        else:
            raise ValueError("Please specify a valid mask type")

    # def _get_all_label(self):
    #     # Just a toy case, please don't use
    #     toy_df = pd.read_csv("/home/tiankang/AU_Dataset/PAIN/master_file_PAIN.csv")
    #     target_df = toy_df.iloc[:,7::].to_numpy()
    #     return target_df


class dataset_BP4D(DataBase):
    """
    BP4D dataset specific
    """

    def __init__(self, master_file, train_mode, mask_type, img_size=256):
        # AU6, AU10, AU12, AU14, AU17

        super().__init__(
            master_file, train_mode, mask_type=mask_type, img_size=img_size
        )
        self._name = "BP4D"
        self._has_AU = True
        self.train_mode = train_mode
        # Several transformation modes
        if train_mode == "train":
            self.transform = image_train(crop_size=self.img_size)
            self.land_transform = land_transform(img_size=self.img_size, flip_reflect=np.loadtxt('/Storage/Projects/AU_Discriminator/code_localGlobal/reflect_49.txt'))
        elif train_mode == "test":
            self.transform = image_test(crop_size=self.img_size)
            self.land_transform = land_transform(img_size=self.img_size, flip_reflect=np.loadtxt('/Storage/Projects/AU_Discriminator/code_localGlobal/reflect_49.txt'))

        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU7",
            "AU10",
            "AU12",
            "AU14",
            "AU15",
            "AU17",
            "AU23",
            "AU24",
        ]

    def __getitem__(self, index):
        old_img_path = self.master_file["aligned_path"].iloc[index]
        land = np.array([float(i) for i in self.master_file['aligned_landmark'].iloc[0].strip('][').split(' ') if i != ''])
        img = Image.open(old_img_path)
        if self.train_mode == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.img_size)
            offset_x = random.randint(0, w - self.img_size)
            flip = random.randint(0, 1)
            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
            if self.land_transform is not None:
                land = self.land_transform(land, flip, offset_x, offset_y)

        elif self.train_mode == 'test':
            w, h = img.size
            offset_y = (h - self.img_size)/2
            offset_x = (w - self.img_size)/2
            if self.transform is not None:
                img = self.transform(img)
            if self.land_transform is not None:
                land = self.land_transform(land, 0, offset_x, offset_y)

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)
        img_AUs = np.array(img_AUs)

        return img, torch.from_numpy(img_AUs), torch.from_numpy(land) 

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_BP4DPlus(DataBase):
    def __init__(self, master_file, train_mode, mask_type, img_size=256):
        # AU6, AU10, AU12, AU14, AU17
        super().__init__(
            master_file, train_mode, mask_type=mask_type, img_size=img_size
        )
        self._name = "BP4D+"
        self._has_AU = True
        # initialize transforms object
        if train_mode == "train":
            self.transform = image_train(crop_size=self.img_size)
            self.land_transform = land_transform(img_size=self.img_size, flip_reflect=np.loadtxt('/Storage/Projects/AU_Discriminator/code_localGlobal/reflect_49.txt'))
        elif train_mode == "test":
            self.transform = image_test(crop_size=self.img_size)
            self.land_transform = land_transform(img_size=self.img_size, flip_reflect=np.loadtxt('/Storage/Projects/AU_Discriminator/code_localGlobal/reflect_49.txt'))

        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU7",
            "AU9",
            "AU10",
            "AU12",
            "AU14",
            "AU15",
            "AU17",
            "AU23",
            "AU24",
        ]

    def __getitem__(self, index):

        old_img_path = self.master_file["aligned_path"].iloc[index]
        land = np.array([float(i) for i in self.master_file['aligned_landmark'].iloc[0].strip('][').split(' ') if i != ''])
        img = Image.open(old_img_path)
        if self.train_mode == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.img_size)
            offset_x = random.randint(0, w - self.img_size)
            flip = random.randint(0, 1)
            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
            if self.land_transform is not None:
                land = self.land_transform(land, flip, offset_x, offset_y)

        elif self.train_mode == 'test':
            w, h = img.size
            offset_y = (h - self.img_size)/2
            offset_x = (w - self.img_size)/2
            if self.transform is not None:
                img = self.transform(img)
            if self.land_transform is not None:
                land = self.land_transform(land, 0, offset_x, offset_y)

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)
        img_AUs = np.array(img_AUs)

        return img, torch.from_numpy(img_AUs), torch.from_numpy(land) 

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_PAIN(DataBase):
    def __init__(self, master_file, train_mode, mask_type):
        # AU6, AU10, AU12, AU14, AU17

        super().__init__(master_file, train_mode, mask_type=mask_type)
        self._name = "PAIN"
        self._has_AU = True
        # initialize transforms object
        if train_mode == "train":
            self.identity_transform = None
            self.resize1 = transforms.Resize(self.img_size)
            self.resize2 = transforms.Resize((56, 56))
            self.transforms = transforms.Compose(
                [
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif train_mode == "extreme":
            self.identity_transform = identity_transform(crop_size=self.img_size)
            self.resize1 = transforms.Resize(self.img_size)
            self.resize2 = transforms.Resize((56, 56))
            self.transforms = transforms.Compose(
                [
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ColorJitter(brightness=0.5, hue=0.3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        elif train_mode == "val":
            self.resize1 = transforms.Resize(self.img_size)
            self.resize2 = transforms.Resize((56, 56))
            self.transforms = transforms.Compose([transforms.ToTensor()])
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            raise ValueError("no transformation specified")

        self.avail_AUs = [
            "AU4",
            "AU6",
            "AU7",
            "AU9",
            "AU10",
            "AU12",
            "AU20",
            "AU25",
            "AU26",
            "AU27",
        ]

    def __getitem__(self, index):

        img_path = self.master_file["aligned_path"].iloc[index]

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)

        img_AUs = np.array(img_AUs)

        if self.train_mode == "extreme":
            img = cv2.imread(img_path)
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
            )
            img = self.identity_transform(img)
            img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            img = Image.open(img_path)

        img = self.resize1(img)
        img = self.transforms(img)
        img = self.normalize(img)

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)

        img_AUs = np.array(img_AUs)

        img_cropped = torch.zeros((1, 3, self.img_size, self.img_size))
        return (img_cropped, img, torch.from_numpy(img_AUs))

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_DISFAPlus(DataBase):
    """
    DISFA+ Dataset loader
    """

    def __init__(self, master_file, train_mode, mask_type, threshold=1):
        # AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
        super().__init__(master_file, train_mode, mask_type=mask_type)
        self._name = "DISFAPlus"
        self._has_AU = True
        self.threshold = threshold
        # initialize transforms object
        if train_mode == "train":
            self.identity_transform = identity_transform(crop_size=self.img_size)
            self.transforms = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # ])
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU5",
            "AU6",
            "AU9",
            "AU12",
            "AU15",
            "AU17",
            "AU20",
            "AU25",
            "AU26",
        ]

    def __getitem__(self, index):

        prefix_paths = "/home/tiankang/DISFA_Inte/outs/"
        if "path" in self.master_file.columns:
            img_path = self.master_file["path"].iloc[index]  # .lstrip('0')
        else:
            img_path = self.master_file["aligned_path"].iloc[index]  # .lstrip('0')
        # if len(stripped_img_name) == 4:
        #     stripped_img_name = '0.jpg'

        # img_path = os.path.join(prefix_paths, self.master_file['subject'].iloc[index],
        #                     self.master_file['session'].iloc[index], stripped_img_name)

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)

        img_AUs = np.array(img_AUs)
        img_AUs_cp = img_AUs

        for i in range(img_AUs_cp.shape[0]):
            if img_AUs[i] == -1:
                continue
            elif img_AUs[i] > self.threshold:
                img_AUs_cp[i] = 1
        # img_AUs = np.where(np.logical_and(img_AUs < self.threshold, img_AUs != -1), 0, 1)
        img = cv2.imread(img_path)
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )

        if self.train_mode == "train":
            img = self.identity_transform(img)
            img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transforms(img)

        img_mask = self.load_mask(img=img, index=index)

        return img, torch.from_numpy(img_AUs_cp), img_mask

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_EmotioNet(DataBase):
    """
    EmotioNet Dataset loader
    """

    def __init__(self, master_file, train_mode, mask_type, threshold=1):
        # AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
        super().__init__(master_file, train_mode, mask_type=mask_type)
        self._name = "Emotionet"
        self._has_AU = True
        self.threshold = threshold
        # initialize transforms object
        if train_mode == "train":
            self.identity_transform = identity_transform(crop_size=self.img_size)
            self.transforms = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # ])
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU5",
            "AU6",
            "AU9",
            "AU12",
            "AU17",
            "AU20",
            "AU25",
            "AU26",
            "AU43",
        ]

    def __getitem__(self, index):

        prefix_paths = "/home/tiankang/DISFA_Inte/outs/"
        if "path" in self.master_file.columns:
            img_path = self.master_file["path"].iloc[index]  # .lstrip('0')
        else:
            img_path = self.master_file["aligned_path"].iloc[index]  # .lstrip('0')
        # if len(stripped_img_name) == 4:
        #     stripped_img_name = '0.jpg'

        # img_path = os.path.join(prefix_paths, self.master_file['subject'].iloc[index],
        #                     self.master_file['session'].iloc[index], stripped_img_name)

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i]].iloc[index])
            else:
                img_AUs.append(-1)

        img_AUs = np.array(img_AUs)
        img_AUs_cp = img_AUs

        for i in range(img_AUs_cp.shape[0]):
            if img_AUs[i] == -1:
                continue
            elif img_AUs[i] == 999:  # This is the missing key value
                img_AUs_cp[i] = 0

        img = cv2.imread(img_path)
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )

        if self.train_mode == "train":
            img = self.identity_transform(img)
            img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transforms(img)

        img_mask = self.load_mask(img=img, index=index)

        return img, torch.from_numpy(img_AUs_cp), img_mask

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_CKPAU(DataBase):
    """
    CK Plus AU Dataset
    """

    def __init__(self, master_file, train_mode, mask_type, threshold=1):
        # AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26
        super().__init__(master_file, train_mode, mask_type=mask_type)
        self._name = "CKPlusAU"
        self._has_AU = True
        self.threshold = threshold
        # initialize transforms object
        if train_mode == "train":
            self.identity_transform = identity_transform(crop_size=self.img_size)
            self.transforms = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # ])
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.avail_AUs = [
            "AU1",
            "AU2",
            "AU4",
            "AU5",
            "AU6",
            "AU7",
            "AU9",
            "AU10",
            "AU11",
            "AU12",
            "AU13",
            "AU14",
            "AU15",
            "AU16",
            "AU17",
            "AU18",
            "AU20",
            "AU21",
            "AU23",
            "AU24",
            "AU25",
            "AU26",
            "AU27",
            "AU28",
            "AU29",
            "AU31",
            "AU34",
            "AU38",
            "AU39",
            "AU43",
        ]

    def __getitem__(self, index):

        prefix_paths = "/home/tiankang/DISFA_Inte/outs/"
        if "path" in self.master_file.columns:
            img_path = self.master_file["path"].iloc[index]  # .lstrip('0')
        else:
            img_path = self.master_file["aligned_path"].iloc[index]  # .lstrip('0')

        img_AUs = []
        for i in range(len(self.all_aus)):
            if self.all_aus[i] in self.avail_AUs:
                img_AUs.append(self.master_file[self.all_aus[i] + ".0"].iloc[index])
            else:
                img_AUs.append(-1)

        img_AUs = np.array(img_AUs)
        img_AUs_cp = img_AUs

        for i in range(img_AUs_cp.shape[0]):
            if img_AUs[i] == -1:
                continue
            elif img_AUs[i] == 999:  # This is the missing key value
                img_AUs_cp[i] = 0

        img = cv2.imread(img_path)
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )

        if self.train_mode == "train":
            img = self.identity_transform(img)
            img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.transforms(img)

        img_mask = self.load_mask(img=img, index=index)

        return img, torch.from_numpy(img_AUs_cp), img_mask

    def _get_all_label(self):
        # Just a toy case, please don't use
        aus_labels = self.master_file[self.avail_AUs]
        return aus_labels.to_numpy()


class dataset_Emo1(DataBase):
    def __init__(self, master_file, train_mode, mask_type):
        # integrates multiple emotion datasets, including
        # ExpW, CKPlus and JAFFE
        super().__init__(master_file, train_mode, mask_type=mask_type)
        self._name = "ExpW, CKPlus and JAFFE"

        # initialize transforms object
        if train_mode == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # ])#,
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),  # ])#,
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.avail_AUs = None
        self._has_Emo = True

    def __getitem__(self, index):

        img_path = self.master_file["Path"].iloc[index]
        emo_label = self.master_file["Label"].to_numpy(dtype=np.int16)[index]

        img = Image.open(img_path)
        img_mask = self.load_mask(img=img, index=index)
        img = self.transforms(img)

        return img, emo_label, img_mask


class dataset_land(DataBase):
    def __init__(self, master_file, train_mode, mask_type):
        # integrates multiple emotion datasets, including
        # ExpW, CKPlus and JAFFE
        super().__init__(master_file, train_mode, mask_type=mask_type)

        self._name = "Landmarks"
        if train_mode == "train":
            self.prefix_path = "/Storage/Data/300W_deng/Train/"
            self.master_file = pd.read_csv(
                "/Storage/Data/300W_deng/Train/300W_train.txt",
                delimiter=r"\s+",
                header=None,
            )
        elif train_mode == "validation":
            self.prefix_path = "/Storage/Data/300W_deng/Validation/"
            self.master_file = pd.read_csv(
                "/Storage/Data/300W_deng/Validation/300W_validation.txt",
                delimiter=r"\s+",
                header=None,
            )
        else:
            self.prefix_path = "/Storage/Data/300W_deng/Test/"
            self.master_file = pd.read_csv(
                "/Storage/Data/300W_deng/Test/300W_test.txt",
                delimiter=r"\s+",
                header=None,
            )

        # # initialize transforms object
        # if train_mode == 'train':
        #     self.transforms = transforms.Compose([
        #                         transforms.Resize(self.img_size),
        #                         transforms.RandomRotation(degrees=10),
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.ToTensor(),#])#,
        #                         transforms.Normalize(
        #                         mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])])
        # else:
        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),  # ])#,
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.avail_AUs = None
        self._has_Emo = False

    def __getitem__(self, index):

        img_path = self.prefix_path + self.master_file.loc[index][0]
        land_label = (
            self.master_file.loc[index][1::]
            .to_numpy(dtype=np.float32)
            .reshape(-1, 2)[7::, :]
        )

        img = Image.open(img_path).convert("RGB")
        old_h, old_w = img.size
        img_mask = self.load_mask(img=img, index=index)

        img = self.transforms(img)
        land_label = land_label * np.array(
            [self.img_size / old_h, self.img_size / old_w]
        )
        land_label = land_label.flatten()
        return img, torch.from_numpy(land_label), img_mask


if __name__ == "__main__":

    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from collections import OrderedDict, defaultdict
    from tqdm import tqdm
    from torch.autograd import Variable
    from torchvision.utils import make_grid, save_image
    device = 'cuda:0'

    bp4d_df = pd.read_csv("/home/txie/AU_Dataset/BP4D/BP4D_Concat_MasterFile_New.csv")
    # bp4dp_df = pd.read_csv(
    #     "/home/tiankang/AU_Dataset/BP4DPlus/BP4D+_Concat_MasterFile.csv"
    # )
    # pain_df = pd.read_csv("/home/tiankang/AU_Dataset/PAIN/master_file_PAIN.csv")

    # disfa_df = pd.read_csv("/Storage/Data/DISFA_/DISFA_main.csv")
    # disfaP_df = pd.read_csv(
    #     "/home/tiankang/AU_Dataset/DISFAPlus/NEWDISFA+_Concat_MasterFile.csv"
    # )

    bp4d_ds = dataset_BP4D(master_file=bp4d_df, train_mode="extreme", mask_type=3)
    # disfa_ds = dataset_DISFAPlus(master_file=disfaP_df, train_mode="train", mask_type=3)

    # a, b, c, d = ckp_ds.__getitem__(index=10)
    bp4d_loader = DataLoader(
        bp4d_ds, batch_size=32, shuffle=False, num_workers=0
    )  # , sampler=sampler_bp4d)
    # disfaP_loader = DataLoader(
    #     disfa_ds, batch_size=32, shuffle=False, num_workers=0
    # )  # ,

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    for batch_idx, (img, labels, _) in enumerate(bp4d_loader):
        img = Variable(img).to(device)
        print("iters!")
        if batch_idx == 50:
            break

    print("finished")
    # emo1_train, emo1_test = train_test_split(emo1_df, test_size=0.2, random_state=1)
    # bp4d_train, bp4d_test = train_test_split(bp4d_df, test_size=0.2, random_state=1)
    # pain_train, pain_test = train_test_split(pain_df, test_size=0.2, random_state=1)
    # bp4d_all = dataset_BP4DPlus(master_file = bp4dp_df, train_mode='test', mask_type=6)
    # # self.dataloaders_external = DataLoader(bp4d_all, batch_size=self.config_train_batch, shuffle=False, num_workers=8)

    # bp4d_raw_train = dataset_BP4D(master_file = bp4d_train, train_mode='train', mask_type=3)
    # bp4d_raw_test = dataset_BP4D(master_file = bp4d_test, train_mode='test', mask_type=6)

    # pain_raw_train = dataset_PAIN(master_file = pain_train, train_mode='train', mask_type=3)
    # pain_raw_test = dataset_PAIN(master_file = pain_test, train_mode='test', mask_type=6)

    # # sampler_emo1 = ImbalancedDatasetSampler_SLML(emo1_raw_train)
    # # sampler_bp4d = ImbalancedDatasetSampler_ML(bp4d_raw_train)
    # # sampler_pain = ImbalancedDatasetSampler_ML(pain_raw_train)

    # dataloaders_train = OrderedDict()
    # # dataloaders_train['emo1'] = DataLoader(emo1_raw_train, batch_size=config_train_batch, shuffle=False, num_workers=0, sampler=sampler_emo1)
    # dataloaders_train['bp4d'] = DataLoader(bp4d_raw_train, batch_size=64, shuffle=False, num_workers=8)#, sampler=sampler_bp4d)
    # dataloaders_train['pain'] = DataLoader(pain_raw_train, batch_size=64, shuffle=False, num_workers=8)#, sampler=sampler_pain)

    # dataloaders_test = DataLoader(bp4d_all, batch_size=64, shuffle=False, num_workers=8)

    # all_labels = {}

    # # for task, task_set in enumerate(dataloaders_train.items()):

    # #     task_train_stats_per_AU_list = defaultdict(dict)
    # #     task_train_statistics_list = []

    # #     print(f'currently training task: {task_set[0]}')
    # #     for batch_index, (img, label, _) in tqdm(enumerate(task_set[1])):
    # #         task_train_statistics_list.append(label)

    # #     all_labels[task_set[0]] = task_train_statistics_list

    # # print('finished')

    # for batch_index, (img, label, _) in tqdm(enumerate(dataloaders_test)):
    #     img = Variable(img).to('cuda')
    #     label = Variable(label).long().to('cuda')
    #     print('finish one iters')

    # pred = self.model(img)
    # loss = self.loss(pred, label)

    # loss_test_total += loss.item()

    # # Debug by building a list of all possible predicted values
    # for aus in range(self.num_aus):
    #     au_realName = self.avail_label_list[aus]
    #     if label[0, aus] != -1:
    #     if label[0, aus] != -1:
    #         if not (au_realName in test_preds):
    #             test_preds[au_realName] = (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5).int().numpy()
    #         else:
    #             test_preds[au_realName] = np.concatenate([test_preds[au_realName],(torch.sigmoid(pred.detach().cpu())[:, aus] > 0.5).int().numpy()])

    #         if not (au_realName in test_labels):
    #             test_labels[au_realName] = label.detach()[:, aus].cpu().numpy()
    #         else:
    #             test_labels[au_realName] = np.concatenate([test_labels[au_realName],label.detach()[:, aus].cpu().numpy()])

    # mySet2 = dataset_land(master_file = ' ', train_mode = 'Train', mask_type = '')
    # mySet = dataset_BP4D(master_root='C:\\users\\hello\\', img_root='D:\\sers\\Her\\',  train_mode='Rrye', transform='jw')
