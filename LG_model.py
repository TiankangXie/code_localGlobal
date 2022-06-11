# train multitask image analysis
# There is something strange inside the code:
# Need to customize loss function too
"""
Exploring if vision transformer can directly predict something about the AU occurence?
Conclusion: it seems to be doing a very good job
"""
from __future__ import print_function
import os
import random
import numpy as np
from numpy.lib.function_base import percentile
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob
from PIL import Image
from itertools import chain
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import collections

# to unzip the datasets
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
import collections
from scikitplot.metrics import plot_confusion_matrix

# from torch.utils.tensorboard import SummaryWriter
from parcellNet4_1 import ResNet
from parcellNet4_2 import get_resnet
# from smallParcellNet import localGlobalNet
from multitask_loader import (
    dataset_BP4D,
    dataset_PAIN,
    dataset_Emo1,
    dataset_land,
    dataset_DISFAPlus,
    dataset_BP4DPlus,
    dataset_EmotioNet,
    dataset_CKPAU,
)
import copy
from utils import (
    align_face_68pts,
    UnNormalize,
    multiclassFocalLoss,
    au_softmax_loss,
    CenterLoss,
    min_max,
    UnNormalize,
    MultilabelBalancedRandomSampler
)
from torchvision.utils import make_grid, save_image
import scikitplot as skplt
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import cv2


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class EarlyStopping(object):
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        metrics = torch.FloatTensor(np.array(metrics))

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def statistics(pred, y, thresh=0.5):
    batch_size = pred.size(0)

    # pred = pred > thresh
    pred = pred.long()

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(batch_size):
        if pred[i] == 1:
            if y[i] == 1:
                TP += 1
            elif y[i] == 0:
                FP += 1
            else:
                assert False
        elif pred[i] == 0:
            if y[i] == 1:
                FN += 1
            elif y[i] == 0:
                TN += 1
            else:
                assert False
        else:
            assert False

    statistics_list = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    return statistics_list


def calc_statistics(statistics_list):
    TP = statistics_list["TP"]
    FP = statistics_list["FP"]
    TN = statistics_list["TN"]
    FN = statistics_list["FN"]

    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-20)
    precision = TP / (TP + FP + 1e-20)
    recall = TP / (TP + FN + 1e-20)
    f1_score = 2 * precision * recall / (precision + recall + 1e-20)

    return f1_score, accuracy, precision, recall


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    else:
        old_list["TP"] += new_list["TP"]
        old_list["FP"] += new_list["FP"]
        old_list["TN"] += new_list["TN"]
        old_list["FN"] += new_list["FN"]

    return old_list


def dice_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)


def au_dice_loss(input, target, weight=None, smooth = 1, size_average=True):
    for i in range(input.size(2)):
        # input is log_softmax, t_input is probability
        t_input = (input[:, 1, i]).exp()
        t_target = (target[:, i]).float()
        t_loss = dice_loss(t_input, t_target, smooth)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def au_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    classify_loss = nn.NLLLoss(size_average=size_average, reduce=reduce)

    for i in range(input.size(2)):
        t_input = input[:, :, i]
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        if weight is not None:
            t_loss = t_loss * weight[i]
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def attention_refine_loss(input, target, size_average=True, reduce=True):
    # loss is averaged over each point in the attention map,
    # note that Eq.(4) in our ECCV paper is to sum all the points,
    # change the value of lambda_refine can remove this difference.
    classify_loss = nn.BCELoss(size_average=size_average, reduce=reduce)

    input = input.view(input.size(0), input.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    for i in range(input.size(1)):
        t_input = input[:, i, :]
        t_target = target[:, i, :]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)
    # sum losses of all AUs
    return loss.sum()


# class dice_loss(nn.Module):
#     def __init__(self, device, weight, smooth=1):
#         super(dice_loss, self).__init__()
#         self.device = device
#         self.smooth = smooth
#         self.weight = weight

#     def __call__(self, outs, labels):
#         smooth = 1.
#         loss = 0.
#         for c in range(outs.shape[-1]):
#             iflat = outs[:, c ].view(-1)
#             tflat = labels[:, c].view(-1)
#             intersection = (iflat * tflat).sum()
            
#             w = self.weight[c]
#             loss += w*(1 - ((2. * intersection + smooth) /
#                                 (iflat.sum() + tflat.sum() + smooth)))
#         return loss

class custom_BCELoss(nn.Module):
    def __init__(self, device):
        super(custom_BCELoss, self).__init__()
        self.criteria = nn.BCEWithLogitsLoss(reduction="none")
        self.criteria2 = nn.NLLLoss(reduction="none")
        self.device = device
        self.m = nn.LogSoftmax(dim=2)

    def __call__(self, outs, labels):
        # See: https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/4
        # idx_selected = []
        # for i in range(labels.shape[1]):
        #     if labels[0][i] == -1:
        #         continue;
        #     else:
        #         idx_selected.append(i)
        # Need to mask out bad index values, default is 9 or 999
        # mask_1 = torch.ones((labels[:, idx_selected].shape)).to(self.device)
        # miss_idc = torch.where(labels[:, idx_selected] == 9)
        # mask_1[miss_idc] = 0
        # miss_idc2 = torch.where(labels[:, idx_selected] == 999)
        # mask_1[miss_idc2] = 0
        # batch_loss = 0
        # CHANGE ME
        # for i in range(labels.shape[0]):
        #     label_row = labels[i:(i+1), :]
        #     masks = torch.where(label_row >= 0.)
        #     label_oneHot = label_row[masks]
        #     label_oneHot = F.one_hot(label_row[masks], num_classes=2)
        #     outs_row = outs[i:(i+1),...]
        #     outs_row = outs_row[masks]
        #     outs_row = torch.unsqueeze(outs_row, 0)
        #     label_oneHot = torch.unsqueeze(label_oneHot, 0)
        #     loss_bce = self.criteria(outs_row, label_oneHot.float())
        #     batch_loss += loss_bce

        mask_1 = torch.ones((labels.shape)).to(self.device)
        miss_idc = torch.where(labels == 9)
        mask_1[miss_idc] = 0
        miss_idc2 = torch.where(labels == 999)
        mask_1[miss_idc2] = 0
        miss_idc3 = torch.where(labels == -1)
        mask_1[miss_idc3] = 0
        loss_bce = self.criteria2(outs, labels.long())
        loss_bce *= mask_1
        return torch.sum(loss_bce)


class normal_BCELoss(nn.Module):
    def __init__(self, weights, device):
        super(normal_BCELoss, self).__init__()
        self.criteria = nn.BCEWithLogitsLoss(
            reduction="none", weight=torch.from_numpy(np.array(weights))
        )
        self.device = device

    def __call__(self, outs, labels):
        mask_1 = torch.ones((labels.shape)).to(self.device)
        miss_idc = torch.where(labels == 9)
        mask_1[miss_idc] = 0
        miss_idc2 = torch.where(labels == 999)
        mask_1[miss_idc2] = 0
        miss_idc3 = torch.where(labels == -1)
        mask_1[miss_idc3] = 0
        loss_bce = self.criteria(outs, labels.float())
        loss_bce *= mask_1
        return torch.sum(loss_bce) / torch.sum(mask_1)


class normal_BCELosswProb(nn.Module):
    def __init__(self, device):
        super(normal_BCELosswProb, self).__init__()
        self.criteria = nn.BCELoss(reduction="none")
        self.device = device

    def __call__(self, outs, labels):
        mask_1 = torch.ones((labels.shape)).to(self.device)
        miss_idc = torch.where(labels == 9)
        mask_1[miss_idc] = 0
        miss_idc2 = torch.where(labels == 999)
        mask_1[miss_idc2] = 0
        miss_idc3 = torch.where(labels == -1)
        mask_1[miss_idc3] = 0
        loss_bce = self.criteria(outs, labels.float())
        loss_bce *= mask_1
        return torch.mean(loss_bce)


class AUEMOModel(nn.Module):
    """
    Main class for training AU classifier
    """

    def __init__(self, configs):

        super(AUEMOModel, self).__init__()

        self.device = configs["device"]
        self.name = "AUModel"
        self.num_workers = configs["num_workers"]
        self.configs = configs
        self.config_train_batch = configs["batch_size"]
        self.n_lands = configs["n_land"]
        self.num_aus = configs["n_aus"]
        self.emo_names = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]
        self.au_names = [
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
        ]
        self.loss_weight = torch.Tensor(configs['weights']).to(self.device)
        self.multi_loss = nn.BCEWithLogitsLoss()
        # self.loss_dice = dice_loss(device=self.device, weight=torch.Tensor(configs["weights"]), smooth=1).to(self.device)
        self.loss_mse = nn.MSELoss()
        self.loss_attention = normal_BCELoss(weights=configs["weights"], device=self.device).to(self.device)

        # generator input: [rgb(3) + landmark(1)]
        # discriminator input: [rgb(3)]
        # model = ResNet(in_chs=3, land_num=49, layers=[2,2,2,2], unit_dim=8, num_classes=self.num_aus, map_size=44, crop_size=176).to(self.device)
        model = get_resnet(in_chs=3, num_lands=49, num_classes=self.num_aus, map_size=44, n_layers=[2,2], crop_size=176).to(self.device)
        self.out_channel_multiplier = 2
        self.add_module("model", model)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        # self.optimizer = optim.SGD(
        #     params=self.model.parameters(),
        #     lr = configs['lr'],
        #     momentum=0.9,
        #     weight_decay=5e-4,
        #     nesterov=True
        # )
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            lr = configs['lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        self.center_loss = CenterLoss(num_classes=2, feat_dim=2048, device=self.device)
        self.beta_center_loss = 0

        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)

        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=configs["step_size"],
            gamma=configs["lr_gamma"],
        )

        self.dataloaders_train = None
        self.dataloaders_val = None
        self.dataloaders_test = None

        self.iteration = 0
        # self.tb = SummaryWriter()
        self.random_state = 1
        self.gen_weights_path = configs["save_path"]

        if not os.path.isdir(self.gen_weights_path):
            os.makedirs(self.gen_weights_path, exist_ok=True)

        self.avail_label_list = []
        self.iters_loss = []
        self.sample_dat = configs["sample_dat"]

        self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.beta1 = 0.2
        self.beta2 = 0.8
        self.beta3 = 0.1
        self.alpha = 0.6

        self.tb = SummaryWriter(log_dir=configs['tb_dir'], comment=configs['tb_comment'])

    def load_data(self, sample_dat=None, validate=True):
        """
        sample_dat: take a sample from the dataset for experimentation, expects 0-1
        """
        # emo1_df = pd.read_csv("/home/tiankang/Emo_integrate/emo_data_cbd.csv")
        bp4d_df = pd.read_csv("/home/txie/AU_Dataset/BP4D/BP4D_Concat_MasterFile_New.csv")
        bp4dp_df = pd.read_csv(
            "/home/txie/AU_Dataset/BP4DPlus/BP4D+_Concat_MasterFile.csv"
        )
        pain_df = pd.read_csv("/home/txie/AU_Dataset/PAIN/master_file_PAIN.csv")
        emotioNet_df = pd.read_csv(
            "/home/txie/AU_Dataset/EmotioNet/EmotioNet_master.csv"
        )
        disfa_df = pd.read_csv("/Storage/Data/DISFA_/DISFA_main.csv")
        disfaP_df = pd.read_csv(
            "/home/txie/AU_Dataset/DISFAPlus/NEWDISFA+_Concat_MasterFile.csv"
        )
        ckp_df = pd.read_csv("/home/txie/AU_Dataset/CKPlus/master_file_CK+.csv")

        if validate:
            if sample_dat:
                a1, a2 = train_test_split(
                    bp4d_df, test_size=sample_dat, stratify=bp4d_df[["subject", "task"]])
                bp4d_df = a2

            data_input = bp4d_df

            data_train = data_input[(data_input['data_split']=='P1') | (data_input['data_split']=='P2')]
            data_test = data_input[data_input['data_split']=='P3']
            train_indices = data_input.index[(data_input['data_split']=='P1') | (data_input['data_split']=='P2')].tolist()
            validation_indices = data_input.index[data_input['data_split']=='P3'].tolist()
                
            img_size = 176
            
            data_raw_train = dataset_BP4D(
                master_file=data_train,
                train_mode='train',
                mask_type=3,
                img_size=img_size
            )
            data_raw_test = dataset_BP4D(
                master_file=data_test, train_mode="test", mask_type=6, img_size=img_size
            )

            self.avail_label_list += data_raw_train.avail_AUs

            data_raw_train.all_aus = self.avail_label_list
            data_raw_test.all_aus = self.avail_label_list

            # TODO:
            dataloaders_train = OrderedDict()
            dataloaders_train["bp4d"] = DataLoader(
                data_raw_train,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True) 

            dataloaders_test = OrderedDict()
            dataloaders_test["bp4d"] = DataLoader(
                data_raw_test,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            ) 

            self.dataloaders_train = dataloaders_train
            self.dataloaders_test = dataloaders_test
        
        else: # Case where we train the entire BP4D dataset
            data_input = bp4d_df
            img_size = 176

            data_raw_train = dataset_BP4D(
                master_file=data_input,
                train_mode='train',
                mask_type=3,
                img_size=img_size
            )

            self.avail_label_list += data_raw_train.avail_AUs
            data_raw_train.all_aus = self.avail_label_list
            # data_raw_test.all_aus = self.avail_label_list

            dataloaders_train = OrderedDict()

            dataloaders_train["bp4d"] = DataLoader(
                data_raw_train,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )  # , sampler=sampler_bp4d)

            self.dataloaders_train = dataloaders_train

        disfaP_all = dataset_DISFAPlus(
            master_file=disfaP_df, train_mode="test", mask_type=6
        )
        bp4dP_all = dataset_BP4DPlus(
            master_file=bp4dp_df, train_mode="test", mask_type=3, img_size=img_size
        )

        disfaP_all.all_aus = self.avail_label_list
        bp4dP_all.all_aus = self.avail_label_list

        self.dataloaders_external = DataLoader(
            bp4dP_all,
            batch_size=self.config_train_batch,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

    def load_data_CV(self, split_mode="P1", validate=True):
        """
        Perform Cross-Validation of the BP4D Dataset
        """
        # emo1_df = pd.read_csv("/home/tiankang/Emo_integrate/emo_data_cbd.csv")
        bp4d_df = pd.read_csv("/home/txie/AU_Dataset/BP4D/BP4D_Concat_MasterFile_New.csv")
        bp4dp_df = pd.read_csv(
            "/home/txie/AU_Dataset/BP4DPlus/BP4D+_Concat_MasterFile.csv"
        )

        data_input = bp4d_df
        img_size = 224

        if validate:
            if split_mode.lower() == "p1":
                data_train = data_input[(data_input['data_split']=='P2') | (data_input['data_split']=='P3')]
                data_test = data_input[data_input['data_split']=='P1']
            elif split_mode.lower() == "p2":
                data_train = data_input[(data_input['data_split']=='P1') | (data_input['data_split']=='P3')]
                data_test = data_input[data_input['data_split']=='P2']
            elif split_mode.lower() == "p3":
                data_train = data_input[(data_input['data_split']=='P1') | (data_input['data_split']=='P2')]
                data_test = data_input[data_input['data_split']=='P3']

            data_raw_train = dataset_BP4D(
                master_file=data_train,
                train_mode=self.transformation,
                mask_type=3,
                img_size=img_size,
            )
            data_raw_test = dataset_BP4D(
                master_file=data_test, train_mode="val", mask_type=6, img_size=img_size
            )
            self.avail_label_list += data_raw_train.avail_AUs

            data_raw_train.all_aus = self.avail_label_list
            data_raw_test.all_aus = self.avail_label_list


            dataloaders_train = OrderedDict()
            dataloaders_train["bp4d"] = DataLoader(
                data_raw_train,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
            ) 

            dataloaders_test = OrderedDict()
            dataloaders_test["bp4d"] = DataLoader(
                data_raw_test,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
            )  

            self.dataloaders_train = dataloaders_train
            self.dataloaders_test = dataloaders_test
        
        else: # Case where we train the entire BP4D dataset
            data_input = bp4d_df[data_input['data_split'] != 'P4']

            img_size = 224

            data_raw_train = dataset_BP4D(
                master_file=data_input,
                train_mode=self.transformation,
                mask_type=3,
                img_size=img_size,
            )

            self.avail_label_list += data_raw_train.avail_AUs

            data_raw_train.all_aus = self.avail_label_list
            data_raw_test.all_aus = self.avail_label_list

            dataloaders_train = OrderedDict()

            dataloaders_train["bp4d"] = DataLoader(
                data_raw_train,
                batch_size=self.config_train_batch,
                shuffle=True,
                num_workers=self.num_workers,
            )  

            self.dataloaders_train = dataloaders_train

        bp4dP_all = dataset_BP4DPlus(
            master_file=bp4dp_df, train_mode="val", mask_type=3, img_size=img_size
        )

        bp4dP_all.all_aus = self.avail_label_list

        self.dataloaders_external = DataLoader(
            bp4dP_all,
            batch_size=self.config_train_batch,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def train_au(self):
        """
        train the network with images, without landmarks
        """
        self.model.train()
        loss_train_total = 0
        loss_train_dice = 0
        loss_train_softmax = 0
        loss_train_land = 0
        loss_train_refine = 0

        train_stats_all = collections.defaultdict(dict)

        dic_items = list(self.dataloaders_train.items())
        random.shuffle(dic_items)

        for task, task_set in enumerate(dic_items):
            # task_train_stats_per_AU_list = collections.defaultdict(dict)
            # task_train_statistics_list = []
            # train_stats_all[task_set[0]] = {}
            task_train_preds = {}
            task_train_labels = {}
            task_train_stats = collections.defaultdict(list)

            print(f"currently training task: {task_set[0]}")
            data_loadinger = tqdm(task_set[1])
            for batch_index, (img, label, land) in enumerate(data_loadinger):
                img = Variable(img).to(self.device, dtype=torch.float)
                label = Variable(label).long().to(self.device)
                land = Variable(land).float().to(self.device)

                # Convert
                self.optimizer.zero_grad()
                x_land, attention_refine, attention_map, pred = self.model.forward(img)

                loss_land = self.loss_mse(x_land, land)
                loss_dice1 = au_dice_loss(input=pred, target=label, weight=self.loss_weight)
                loss_softmax = au_softmax_loss(input=pred, target=label, weight=self.loss_weight)
                resized_aus_map = F.interpolate(attention_map, size=attention_refine.size(2))
                loss_refine = attention_refine_loss(input=attention_refine, target=resized_aus_map)
                
                total_loss = loss_land + loss_softmax + loss_dice1 + loss_refine

                loss_train_total += total_loss.item()
                loss_train_dice += loss_dice1.item()
                loss_train_softmax += loss_softmax.item()
                loss_train_land += loss_land.item()
                loss_train_refine += loss_refine.item()

                total_loss.backward()
                self.optimizer.step()

                data_loadinger.set_description(
                    "[EPOCH %d] [BATCH %d]: | TOTAL:%.3f | LAND:%.3f | SOFTMAX:%.3f | DICE: %.3f | REFINE:%.3f"
                    % (self.iteration, batch_index, total_loss.item(), loss_land.item(), loss_softmax.item(), loss_dice1.item(), loss_refine.item())
                )

                if batch_index == 150:
                    n_img_show = 12
                    for n_au_plot in range(attention_refine.shape[1]):
                        c_att = attention_refine[n_img_show, n_au_plot, :, :].detach().cpu().numpy()
                        resize_att = cv2.resize(c_att, (176, 176))
                        heatmap = cv2.applyColorMap(np.uint8(255 * resize_att), cv2.COLORMAP_JET)
                        # if use_rgb:
                        #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = np.float32(heatmap) / 255

                        v_img = self.unorm(img[n_img_show]).permute((1,2,0)).detach().cpu().numpy()
                        v_img = v_img[:, :, ::-1]
                        cam = heatmap + v_img
                        cam = cam / np.max(cam)
                        fused_img = np.uint8(255 * cam)
                        self.tb.add_image(f'AU{self.avail_label_list[n_au_plot]}/train', np.transpose(fused_img, (2, 0, 1)), self.iteration)

                        # Also the raw attention layer
                        c_att = attention_map[n_img_show, n_au_plot, :, :].detach().cpu().numpy()
                        resize_att = cv2.resize(c_att, (176, 176))
                        heatmap = cv2.applyColorMap(np.uint8(255 * resize_att), cv2.COLORMAP_JET)
                        # if use_rgb:
                        #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = np.float32(heatmap) / 255

                        v_img = self.unorm(img[n_img_show]).permute((1,2,0)).detach().cpu().numpy()
                        v_img = v_img[:, :, ::-1]
                        cam = heatmap + v_img
                        cam = cam / np.max(cam)
                        fused_img = np.uint8(255 * cam)
                        self.tb.add_image(f'Att_Map_AU{self.avail_label_list[n_au_plot]}/train', np.transpose(fused_img, (2, 0, 1)), self.iteration)
                #################################SIGMOID############################################################
                # Debug by building a list of all possible predicted values
                if self.out_channel_multiplier == 1:
                    for aus in range(len(self.avail_label_list)):
                        au_realName = self.avail_label_list[
                            aus
                        ]  # Real Name for Action Units
                        if label[0, aus] != -1:
                            if False:
                                pred_val = (
                                    (pred.detach().cpu()[:, aus] > 0.5).int().numpy()
                                )
                            else:
                                pred_val = (
                                    (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5)
                                    .int()
                                    .numpy()
                                )

                            if not (au_realName in task_train_preds):
                                task_train_preds[au_realName] = pred_val
                            else:
                                task_train_preds[au_realName] = np.concatenate(
                                    [task_train_preds[au_realName], pred_val]
                                )

                            if not (au_realName in task_train_labels):
                                task_train_labels[au_realName] = (
                                    label.detach()[:, aus].cpu().numpy()
                                )
                            else:
                                task_train_labels[au_realName] = np.concatenate(
                                    [
                                        task_train_labels[au_realName],
                                        label.detach()[:, aus].cpu().numpy(),
                                    ]
                                )
                ##################################SOFTMAX############################################################
                elif self.out_channel_multiplier == 2:
                    for aus in range(len(self.avail_label_list)):
                        au_realName = self.avail_label_list[
                            aus
                        ]  # Real Name for Action Units
                        if label[0, aus] != -1:
                            if not (au_realName in task_train_preds):
                                task_train_preds[au_realName] = (torch.argmax(pred.detach().cpu(), dim=1).int().numpy()[:, aus])
                            else:
                                task_train_preds[au_realName] = np.concatenate(
                                    [
                                        task_train_preds[au_realName],
                                        torch.argmax(pred.detach().cpu(), dim=1).int().numpy()[:, aus],
                                    ]
                                )

                            if not (au_realName in task_train_labels):
                                task_train_labels[au_realName] = (
                                    label.detach()[:, aus].cpu().numpy()
                                )
                            else:
                                task_train_labels[au_realName] = np.concatenate(
                                    [
                                        task_train_labels[au_realName],
                                        label.detach()[:, aus].cpu().numpy(),
                                    ]
                                )
            print("finished preprocessing")

            for aus in task_train_preds.keys():
                au_label_vals = task_train_labels[aus]
                au_label_valid_idx = np.where(
                    np.logical_or(au_label_vals == 0, au_label_vals == 1)
                )
                au_pred_vals = task_train_preds[aus]
                (
                    au_spec_prec,
                    au_spec_rec,
                    au_spec_fsco,
                    au_spec_supp,
                ) = precision_recall_fscore_support(
                    au_label_vals[au_label_valid_idx],
                    au_pred_vals[au_label_valid_idx],
                    pos_label=1,
                    average="binary",
                )
                task_train_stats[aus] = [
                    au_spec_prec,
                    au_spec_rec,
                    au_spec_fsco,
                    au_spec_supp,
                ]
            # au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp = precision_recall_fscore_support(task_train_preds[aus], task_train_labels[aus], pos_label=1, average='micro')
            # task_train_stats[aus] = [au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp]
            train_stats_all[task_set[0]] = task_train_stats
            # # print task specific values

        print("Overall training accuracies")
        for kk in train_stats_all.keys():
            print(f"For dataset: {kk}")
            for kk2 in train_stats_all[kk].keys():
                print(
                    f"AU: {kk2}, f1 score: {train_stats_all[kk][kk2][2]}, precision is: {train_stats_all[kk][kk2][0]}, recall is: {train_stats_all[kk][kk2][1]}"
                )
                self.tb.add_scalar(f'AU{kk2} F1/train', train_stats_all[kk][kk2][2], self.iteration)

        # Start training objects
        self.tb.add_scalar('Total Loss/train', loss_train_total, self.iteration)
        self.tb.add_scalar('Total Dice/train', loss_train_dice, self.iteration)
        self.tb.add_scalar('Total Softmax/train', loss_train_softmax, self.iteration)
        self.tb.add_scalar('Total Refine/train', loss_train_refine, self.iteration)
        self.tb.add_scalar('Total Land/train', loss_train_land, self.iteration)

        print(f"Total Training loss is: {loss_train_total}")
        self.iters_loss.append(loss_train_total)
        # self.tb.add_scalar('Total L2 Loss/train', total_l2_loss/(total_number*self.n_lands), self.iteration)
        # report_progress(total_gan_loss, total_dis_loss, self.iteration)
        # print(f'Loss function for iters/train: {self.iteration},: {total_l2_loss/(total_number*self.n_lands)}')

    def validate(self):
        """
        train the network with images, without landmarks
        """
        self.model.eval()

        with torch.no_grad():

            val_stats_all = collections.defaultdict(dict)

            for task, task_set in enumerate(self.dataloaders_test.items()):

                task_losses = {}
                task_val_preds = {}
                task_val_labels = {}
                task_val_stats = collections.defaultdict(list)
                
                loss_val_total = 0
                loss_val_dice = 0
                loss_val_softmax = 0
                loss_val_land = 0
                loss_val_refine = 0
                batch_img_size = 0

                print(f"currently validating task: {task_set[0]}")
                data_loadinger = tqdm(task_set[1])
                for batch_index, (img, label, land) in enumerate(data_loadinger):
                    batch_img_size += img.shape[0]
                    img = Variable(img).to(self.device, dtype=torch.float)
                    label = Variable(label).long().to(self.device)
                    land = Variable(land).float().to(self.device)

                    # Convert
                    x_land, attention_refine, attention_map, pred = self.model(img)
                    loss_land = self.loss_mse(x_land, land)
                    loss_dice1 = au_dice_loss(input=pred, target=label, weight=self.loss_weight)
                    loss_softmax = au_softmax_loss(input=pred, target=label, weight=self.loss_weight)
                    resized_aus_map = F.interpolate(attention_map, size=attention_refine.size(2))
                    loss_refine = attention_refine_loss(input=attention_refine, target=resized_aus_map)
                    
                    total_loss = loss_land + loss_softmax + loss_dice1 + loss_refine

                    loss_val_total += total_loss.item()
                    loss_val_dice += loss_dice1.item()
                    loss_val_softmax += loss_softmax.item()
                    loss_val_land += loss_land.item()
                    loss_val_refine += loss_refine.item()

                    data_loadinger.set_description(
                        "[EPOCH %d] [BATCH %d]: | TOTAL:%.3f | LAND:%.3f | SOFTMAX:%.3f | DICE: %.3f | REFINE:%.3f"
                        % (self.iteration, batch_index, total_loss.item(), loss_land.item(), loss_softmax.item(), loss_dice1.item(), loss_refine.item())
                    )

                    if batch_index == 25:
                        n_img_show = 12
                        for n_au_plot in range(attention_refine.shape[1]):
                            c_att = attention_refine[n_img_show, n_au_plot, :, :].detach().cpu().numpy()
                            resize_att = cv2.resize(c_att, (176, 176))
                            heatmap = cv2.applyColorMap(np.uint8(255 * resize_att), cv2.COLORMAP_JET)
                            # if use_rgb:
                            #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            heatmap = np.float32(heatmap) / 255

                            v_img = self.unorm(img[n_img_show]).permute((1,2,0)).detach().cpu().numpy()
                            v_img = v_img[:, :, ::-1]
                            cam = heatmap + v_img
                            cam = cam / np.max(cam)
                            fused_img = np.uint8(255 * cam)
                            self.tb.add_image(f'AU{self.avail_label_list[n_au_plot]}/val', np.transpose(fused_img, (2, 0, 1)), self.iteration)
                    ##################################SIGMOID############################################################
                    # Debug by building a list of all possible predicted values
                    if self.out_channel_multiplier == 1:
                        for aus in range(len(self.avail_label_list)):
                            au_realName = self.avail_label_list[aus]
                            if label[0, aus] != -1:
                                if False:
                                    pred_val = (
                                        (pred.detach().cpu()[:, aus] > 0.5)
                                        .int()
                                        .numpy()
                                    )
                                else:
                                    pred_val = (
                                        (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5)
                                        .int()
                                        .numpy()
                                    )

                                if not (au_realName in task_val_preds):
                                    task_val_preds[au_realName] = pred_val
                                else:
                                    task_val_preds[au_realName] = np.concatenate(
                                        [task_val_preds[au_realName], pred_val]
                                    )

                                if not (au_realName in task_val_labels):
                                    task_val_labels[au_realName] = (
                                        label.detach()[:, aus].cpu().numpy()
                                    )
                                else:
                                    task_val_labels[au_realName] = np.concatenate(
                                        [
                                            task_val_labels[au_realName],
                                            label.detach()[:, aus].cpu().numpy(),
                                        ]
                                    )
                    ##################################SOFTMAX############################################################
                    elif self.out_channel_multiplier == 2:
                        for aus in range(len(self.avail_label_list)):
                            au_realName = self.avail_label_list[
                                aus
                            ]  # Real Name for Action Units
                            if label[0, aus] != -1:
                                if not (au_realName in task_val_preds):
                                    task_val_preds[au_realName] = (
                                        torch.argmax(pred.detach().cpu(), dim=1)
                                        .int()
                                        .numpy()[:, aus]
                                    )
                                else:
                                    task_val_preds[au_realName] = np.concatenate(
                                        [
                                            task_val_preds[au_realName],
                                            torch.argmax(pred.detach().cpu(), dim=1)
                                            .int()
                                            .numpy()[:, aus],
                                        ]
                                    )

                                if not (au_realName in task_val_labels):
                                    task_val_labels[au_realName] = (
                                        label.detach()[:, aus].cpu().numpy()
                                    )
                                else:
                                    task_val_labels[au_realName] = np.concatenate(
                                        [
                                            task_val_labels[au_realName],
                                            label.detach()[:, aus].cpu().numpy(),
                                        ]
                                    )
                            # statistics_list = statistics((F.sigmoid(pred.detach())[:, aus] > 0.5).int(), label.detach()[:, aus], 0.5)
                            # train_statistics_list = update_statistics_list(train_statistics_list, statistics_list)
                            # task_train_statistics_list = update_statistics_list(task_train_statistics_list, statistics_list)
                            # task_train_stats_per_AU_list[aus] = copy.deepcopy(task_train_statistics_list)
                            # train_stats_per_AU_list[aus] = copy.deepcopy(train_statistics_list)

                print("finished validation preprocessing")

                for aus in task_val_preds.keys():
                    au_label_vals = task_val_labels[aus]
                    au_label_valid_idx = np.where(
                        np.logical_or(au_label_vals == 0, au_label_vals == 1)
                    )
                    au_pred_vals = task_val_preds[aus]
                    (
                        au_spec_prec,
                        au_spec_rec,
                        au_spec_fsco,
                        au_spec_supp,
                    ) = precision_recall_fscore_support(
                        au_label_vals[au_label_valid_idx],
                        au_pred_vals[au_label_valid_idx],
                        pos_label=1,
                        average="binary",
                    )
                    task_val_stats[aus] = [
                        au_spec_prec,
                        au_spec_rec,
                        au_spec_fsco,
                        au_spec_supp,
                    ]
                # au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp = precision_recall_fscore_support(task_train_preds[aus], task_train_labels[aus], pos_label=1, average='micro')
                # task_train_stats[aus] = [au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp]
                val_stats_all[task_set[0]] = task_val_stats
                task_losses[task_set[0]] = loss_val_total / batch_img_size
                # # print task specific values
                # if task == 0:
                #     class_rep = classification_report(task_train_labels_emo, task_preds_labels_emo, target_names=target_names)
                #     emo_train_f1 = f1_score(task_train_labels_emo, task_preds_labels_emo, average='macro')
                #     print(class_rep)

                # for ttsk in range(len(list(task_train_stats_per_AU_list.keys()))):
                #     kkey = list(task_train_stats_per_AU_list.keys())[ttsk]
                #     print('for action unit:')
                #     print(self.au_names[kkey]) # minus 2 to align with the original index
                #     # print(task_train_stats_per_AU_list[ttsk])
                #     au_train_f1, au_train_acc, au_train_pr, au_train_re = calc_statistics(task_train_stats_per_AU_list[kkey])
                #     print(au_train_f1, au_train_acc, au_train_pr, au_train_re)
                #     # plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))

            print("Overall validation accuracies")
            for kk in val_stats_all.keys():
                print(f"For dataset: {kk}")
                for kk2 in val_stats_all[kk].keys():
                    print(
                        f"AU: {kk2}, f1 score is: {val_stats_all[kk][kk2][2]}, precision is: {val_stats_all[kk][kk2][0]}, recall is: {val_stats_all[kk][kk2][1]}"
                    )
                    self.tb.add_scalar(f'AU{kk2} F1/val', val_stats_all[kk][kk2][2], self.iteration)

            print('avg test f1 is:', np.mean([mm[2] for mm in list(val_stats_all['bp4d'].values())]))
            # au_train_f1, au_train_acc, au_train_pr, au_train_re = calc_statistics(train_statistics_list)
            # print(au_train_f1, au_train_acc, au_train_pr, au_train_re)
            print(f"Total Validation loss is: {loss_val_total}")

            self.tb.add_scalar('Total Loss/val', loss_val_total, self.iteration)
            self.tb.add_scalar('Total Dice/val', loss_val_dice, self.iteration)
            self.tb.add_scalar('Total Softmax/val', loss_val_softmax, self.iteration)
            self.tb.add_scalar('Total Refine/val', loss_val_refine, self.iteration)
            self.tb.add_scalar('Total Land/val', loss_val_land, self.iteration)
            # Start training objects
            # print('training on dataset')
            # self.tb.add_scalar('Total L2 Loss/train', total_l2_loss/(total_number*self.n_lands), self.iteration)
            # report_progress(total_gan_loss, total_dis_loss, self.iteration)
            # print(f'Loss function for iters/train: {self.iteration},: {total_l2_loss/(total_number*self.n_lands)}')

    def load_pretrained(self, path_name="/Storage/Projects/lafin_img/results/"):

        self.model.load_state_dict(
            torch.load(path_name, map_location=self.device)["generator"]
        )

    def test(self, data_loaders, debug_path=None, debug=False):
        """
        test performance on another dataset
        images: raw images
        masks: raw masks
        """

        print("Start testing with external dataset")
        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.model.eval()
        with torch.no_grad():
            loss_test_total = 0
            test_preds = {}
            test_labels = {}
            test_stats = collections.defaultdict(list)
            # data_loaders = tqdm(data_loaders)
            # print(f'currently testing: {task_set[0]}')
            for batch_index, (img, label, lands) in enumerate(tqdm(data_loaders)):

                img = Variable(img).to(self.device, dtype=torch.float)
                label = Variable(label).long().to(self.device)

                x_land, attention_refine, attention_map, pred = self.model(img)
                # loss_test_total += loss.item()
                # n_img_show = 4
                # for n_au_plot in range(len(attentions)):
                #     c_att = attentions[n_au_plot][n_img_show, :, :, :].permute((1,2,0)).detach().cpu().numpy()
                #     resize_att = cv2.resize(c_att, (224, 224))
                #     resize_att = min_max(resize_att, axis=1)
                #     resize_att *= (255)

                #     v_img = self.unorm(img[n_img_show]).permute((1,2,0)).detach().cpu().numpy() * 255
                #     v_img = v_img[:, :, ::-1]
                #     v_img = np.uint8(v_img)
                #     vis_map = np.uint8(resize_att)
                #     jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
                #     jet_map = cv2.add(v_img, jet_map)
                #     cv2.imwrite(f'/Storage/Projects/FER_GAN/code/faceParsing/test_img/result_img/testing1_{self.avail_label_list[n_au_plot]}.jpg', jet_map)
                #     cv2.imwrite(f'/Storage/Projects/FER_GAN/code/faceParsing/test_img/result_img/testing2_{self.avail_label_list[n_au_plot]}.jpg', v_img)

                recorded_AU = []
                recorded_f1 = []
                recorded_acc = []
                for aus in range(len(self.avail_label_list)):
                    au_realName = self.avail_label_list[aus]

                    if label[0, aus] != -1:
                        # if self.model_type.lower() == 'abdnetwork':
                        #     pred_val = (pred.detach().cpu()[:, aus] > 0.5).int().numpy()
                        # elif self.model_type.lower() == 'abdnetwork2':
                        #     pred_val = torch.argmax(pred.detach().cpu(), dim=1).int().numpy()[:,aus]
                        # else:
                        #     pred_val = (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5).int().numpy()

                        if self.out_channel_multiplier == 1:
                            if "abdnetwork" in self.model_type.lower():
                                tmp_pred = (
                                    (pred.detach().cpu()[:, aus] > 0.5).int().numpy()
                                )
                            else:
                                if True:
                                    tmp_pred = (
                                        F.sigmoid(pred.detach().cpu())[:, aus]
                                    ).numpy()
                                else:
                                    tmp_pred = (
                                        (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5)
                                        .int()
                                        .numpy()
                                    )

                        elif self.out_channel_multiplier == 2:
                            tmp_pred = (
                                torch.argmax(pred.detach().cpu(), dim=1)
                                .int()
                                .numpy()[:, aus]
                            )

                        if not (au_realName in test_preds):
                            test_preds[au_realName] = tmp_pred
                        else:
                            test_preds[au_realName] = np.concatenate(
                                [test_preds[au_realName], tmp_pred]
                            )

                        tmp_label = label.detach()[:, aus].cpu().numpy()
                        if not (au_realName in test_labels):
                            test_labels[au_realName] = tmp_label
                        else:
                            test_labels[au_realName] = np.concatenate(
                                [test_labels[au_realName], tmp_label]
                            )

                        # tmp_label_mask = np.where(np.logical_or(tmp_label==0, tmp_label==1))
                        # _, _, tmp_au_fsco, _ = precision_recall_fscore_support(tmp_pred[tmp_label_mask], tmp_label[tmp_label_mask], pos_label=1, average='binary')
                        # tmp_au_acc = accuracy_score(y_true=tmp_label[tmp_label_mask], y_pred=tmp_pred[tmp_label_mask])
                        # recorded_AU.append(au_realName)
                        # recorded_f1.append(tmp_au_fsco)
                        # recorded_acc.append(tmp_au_acc)
                        if debug:
                            import os

                            mypath = debug_path + au_realName
                            if not os.path.isdir(mypath):
                                os.makedirs(mypath)

                            mypath_pos = mypath + "/" + "label_pos"
                            if not os.path.isdir(mypath_pos):
                                os.makedirs(mypath_pos)

                            img_debug = (
                                img[
                                    np.where(
                                        np.logical_and(tmp_pred == 0, tmp_label == 1)
                                    )
                                ]
                                .detach()
                                .cpu()
                            )
                            if img_debug.size(0) > 0:
                                save_image(
                                    [
                                        unorm(img_debug[mm][0:3])
                                        for mm in range(img_debug.size(0))
                                    ],
                                    fp=mypath_pos + f"/diag_plot{batch_index}.jpg",
                                )

                            mypath_neg = mypath + "/" + "label_neg"
                            if not os.path.isdir(mypath_neg):
                                os.makedirs(mypath_neg)

                            img_debug = (
                                img[
                                    np.where(
                                        np.logical_and(tmp_pred == 1, tmp_label == 0)
                                    )
                                ]
                                .detach()
                                .cpu()
                            )
                            if img_debug.size(0) > 0:
                                save_image(
                                    [
                                        unorm(img_debug[mm][0:3])
                                        for mm in range(img_debug.size(0))
                                    ],
                                    fp=mypath_neg + f"/diag_plot{batch_index}.jpg",
                                )
                # data_loaders.set_description(
                #     f'[BATCH {batch_index}]: | LOSS:{loss.item()} | AU1 F1 {recorded_f1[0]}: | AU1 Acc:{recorded_acc[0]} \
                #     | AU2 F1:{recorded_f1[1]} | AU2 Acc:{recorded_acc[1]} | AU4 F1:{recorded_f1[2]} | AU4 Acc:{recorded_acc[2]} \
                #     | AU6 F1:{recorded_f1[3]} | AU6 Acc:{recorded_acc[3]} | AU7 F1:{recorded_f1[4]} | AU7 Acc:{recorded_acc[4]}')
        print("finished preprocessing")

        for aus in test_preds.keys():
            
            au_label_vals = test_labels[aus]
            au_pred_vals = test_preds[aus]
            valid_indx = np.where((au_label_vals==0) | (au_label_vals==1))

            au_spec_prec,au_spec_rec,au_spec_fsco,au_spec_supp = precision_recall_fscore_support( \
                                au_pred_vals[valid_indx],
                                au_label_vals[valid_indx],
                                pos_label=1,
                                average="binary")
            test_stats[aus] = [
                au_spec_prec,
                au_spec_rec,
                au_spec_fsco,
                au_spec_supp,
            ]

            self.tb.add_scalar(f'AU{aus} F1/test', au_spec_fsco, self.iteration)

        print(f'avg test f1 is:{np.mean([mm[2] for mm in list(test_stats.values())])}')

    def resume_training(self, check_iters):

        checkpoint_gen = torch.load(
            os.path.join(self.gen_weights_path, self.name + f"_{check_iters}.pth")
        )
        self.model.load_state_dict(checkpoint_gen["generator"])
        self.optimizer.load_state_dict(checkpoint_gen["optimizer"])
        self.iteration = checkpoint_gen["iteration"]
        print(f"resuming training from iters:{check_iters}")

    def save(self):
        # Save per iteration or
        print("\nsaving %s...\n" % self.name)
        torch.save(
            {
                "iteration": self.iteration,
                "generator": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            os.path.join(self.gen_weights_path, self.name + f"_{self.iteration}.pth"),
        )

    def load_ckp(self, check_iters):
        """
        Continue training by loading the default data
        """
        checkpoint_gen = torch.load(
            os.path.join(self.gen_weights_path, self.name + f"_{check_iters}.pth")
        )
        checkpoint_discr = torch.load(
            self.gen_weights_path + self.name + f"_{check_iters}.pth"
        )

        self.model.load_state_dict(checkpoint_gen["generator"])
        # self.discriminator.load_state_dict(checkpoint_discr['discriminator'])
        # self.gen_optimizer.load_state_dict(checkpoint_gen['optimizer'])
        # self.dis_optimizer.load_state_dict(checkpoint_discr['optimizer'])

        self.iteration = checkpoint_gen["iteration"]

    def run_train(self, num_epochs, resume_training=False, validate=True,resume_from_iters=10):

        # Load data
        if not self.dataloaders_train:
            self.load_data(sample_dat=self.sample_dat, validate=validate)

        if resume_training:
            print(f"resuming from iters {resume_from_iters}")
            self.resume_training(resume_from_iters)

        while self.iteration <= num_epochs:
            print(f"======Loss for epoch num {self.iteration}")
            print('learning rate:', self.optimizer.param_groups[0]['lr'])
            self.train_au()
            if validate:
                self.validate()
            else:
                self.test(
                    data_loaders=self.dataloaders_external, debug_path=None, debug=False)

            self.iteration += 1

            if self.iteration >= self.configs["start_save"]:
                self.save()
                self.scheduler.step()

        # self.test(data_loaders=self.dataloaders_external)
        # print('=============================================')
        # for iters in [2, 6, 10, 16, 20, 24]:
        #     print(f'=============={iters}==============')
        #     self.run_test(iteration_load=iters, debug_path='/Storage/Projects/FER_GAN/code/faceParsing/test_img/',debug=False)
        # self.test(data_loaders=self.dataloaders_external2)
        self.tb.close()
        return

    def run_train_CV(self, num_epochs, split_data='P1'):

        # Load data
        if not self.dataloaders_train:
            self.load_data_CV(split_mode=split_data, validate=True)

        while self.iteration <= num_epochs:
            print(f"======Loss for epoch num {self.iteration}")
            self.train_au()
            self.validate()
            if self.iteration > 0 and self.iteration % 2 == 0:
                self.test(
                    data_loaders=self.dataloaders_external, debug_path=None, debug=False
                )

            if self.iteration >= self.configs["start_save"]:
                # self.save()
                self.scheduler.step()
            self.iteration += 1

        self.tb.close()
        return

    def run_test(self, iteration_load, debug_path=None, debug=False):

        # Load data
        if not self.dataloaders_train:
            self.load_data(sample_dat=self.sample_dat)

        self.load_ckp(check_iters=iteration_load)
        self.test(
            data_loaders=self.dataloaders_external, debug_path=debug_path, debug=debug
        )
        # self.test(data_loaders=self.dataloaders_external2, debug=debug)
        return

    def run_test_fromTrain(self, num_epochs):
        # Testing external data by running over the entire training set. 
        # Load data
        if not self.dataloaders_train:
            self.load_data(sample_dat=self.sample_dat, validate=True)

        while self.iteration <= num_epochs:
            print(f"======Loss for epoch num {self.iteration}")
            self.train_au()
            self.test(data_loaders=self.dataloaders_external, debug_path=None, debug=False)
            if self.iteration >= self.configs["start_save"]:
                self.scheduler.step()
            self.iteration += 1
        return

    def run_validation(self, iteration_load, debug_path=None, debug=False):

        # Load data x2
        if not self.dataloaders_train:
            self.load_data(sample_dat=self.sample_dat)
        self.load_ckp(check_iters=iteration_load)
        self.validate()
        return


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y : mask_y + mask_height, mask_x : mask_x + mask_width] = 1
    return mask


if __name__ == "__main__":
    import cv2

    """
    Test and get some of the results 
    """
    beta_coeffs = 0.5
    save_dir = "/Storage/Projects/FER_GAN/code/simpleAU/ckpt_bp4d_ParcellLG_ATTFuse/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    configs = {
        "n_land": 68,
        "in_chs": 0 + 3,
        "n_aus": 13,
        "start_save": 2,
        "num_workers": 16,
        "step_size": 4,
        "lr_gamma": 0.2,
        "batch_size": 64,
        "lr": 1e-4,
        "save_path": save_dir,
        "model_type": "localglobal_gcn",
        "extreme_transform": "train",
        "weights": [
            7.9420,
            9.0383,
            11.4478,
            1.9043,
            1.4496,
            15.3781,
            1.4818,
            1.6515,
            1.5934,
            7.3480,
            6.2880,
            5.1076,
            14.6253,
        ],
        "data_split_method": "subject",
        "device": "cuda:0",
        "loss": "bce",
        "weight_decay": 4e-5,
        "sample_dat": 0.8,
        "beta": beta_coeffs,
        "num_cuts": 4,
        "experimental_mode": False,
        "share_weight": True,
    }

    inpaint_model = AUEMOModel(configs=configs)
    # inpaint_model.run_validation(iteration_load=6, debug_path='/Storage/Projects/FER_GAN/code/faceParsing/test_img/',debug=False)
    # inpaint_model.run_train(24, resume_training=True, resume_from_iters=10)
    inpaint_model.run_test(
        iteration_load=20,
        debug_path="/Storage/Projects/FER_GAN/code/faceParsing/test_img/",
        debug=False,
    )
    # for iters in [2, 3, 4, 5, 6, 8, 10]:
    #     print(f'=============={iters}==============')
    #     inpaint_model.run_test(iteration_load=iters, debug_path='/Storage/Projects/FER_GAN/code/faceParsing/test_img/', debug=False)
