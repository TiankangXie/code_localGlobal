import sys
import os
import ray
from ray import available_resources, tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
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
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from parcellNet3_1 import localGlobalNet
from parcellNet4 import localGlobalNet as localGlobalNet_adaptive
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import collections
import random
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter


class dice_loss(nn.Module):
    def __init__(self, device, weight, smooth=1):
        super(dice_loss, self).__init__()
        self.device = device
        self.smooth = smooth
        self.weight = weight

    def __call__(self, outs, labels):
        loss = 0.
        for c in range(outs.shape[-1]):
            iflat = outs[:, c ].view(-1)
            tflat = labels[:, c].view(-1)
            intersection = (iflat * tflat).sum()
            
            w = self.weight[c]
            loss += w*(1 - ((2. * intersection + self.smooth) /
                                (iflat.sum() + tflat.sum() + self.smooth)))
        return loss


class custom_BCELoss(nn.Module):
    def __init__(self, device):
        super(custom_BCELoss, self).__init__()
        self.criteria = nn.BCEWithLogitsLoss(reduction="none")
        self.criteria2 = nn.NLLLoss(reduction="none")
        self.device = device
        self.m = nn.LogSoftmax(dim=2)

    def __call__(self, outs, labels):
        # See: https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/4
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
    def __init__(self, weights, device, reduction='sum'):
        super(normal_BCELoss, self).__init__()
        self.criteria = nn.BCEWithLogitsLoss(
            reduction="none", weight=torch.from_numpy(np.array(weights))
        )
        self.device = device
        self.reduction = reduction

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
        if self.reduction == 'sum':
            return torch.sum(loss_bce)
        else:
            return loss_bce


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


def load_data(sample_dat=None, batch_size=48, train_mode='train', split="random"):
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

    # if sample_dat:
    #     a1, a2 = train_test_split(
    #         bp4d_df, test_size=sample_dat, stratify=bp4d_df[["subject", "task"]]
    #     )
    #     bp4d_df = a2

    #     a1, a2 = train_test_split(
    #         bp4dp_df, test_size=sample_dat, stratify=bp4dp_df[["subject", "task"]]
    #     )
    #     bp4dp_df = a2

    data_input = bp4d_df
    if split.lower() == "random":
        data_train, data_test = train_test_split(
            data_input, test_size=(1 / 3), random_state=1
        )
        ckp_ds = dataset_CKPAU(master_file=ckp_df, train_mode=True, mask_type=6)

    elif split.lower() == "subject":
        data_train = data_input[(data_input['data_split']=='P1') | (data_input['data_split']=='P2')]
        data_test = data_input[data_input['data_split']=='P3']

    img_size = 224

    data_raw_train = dataset_BP4D(
        master_file=data_train,
        train_mode='extreme',
        mask_type=3,
        img_size=img_size,
    )
    data_raw_test = dataset_BP4D(
        master_file=data_test, train_mode="val", mask_type=6, img_size=img_size
    )

    avail_label_list = []
    avail_label_list += data_raw_train.avail_AUs

    data_raw_train.all_aus = avail_label_list
    data_raw_test.all_aus = avail_label_list

    dataloaders_train = OrderedDict()
    dataloaders_train["bp4d"] = DataLoader(
        data_raw_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    ) 

    dataloaders_test = OrderedDict()
    dataloaders_test["bp4d"] = DataLoader(
        data_raw_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    disfaP_all = dataset_DISFAPlus(
        master_file=disfaP_df, train_mode="test", mask_type=6
    )
    bp4dP_all = dataset_BP4DPlus(
        master_file=bp4dp_df, train_mode="val", mask_type=3, img_size=img_size
    )
    
    disfaP_all.all_aus = avail_label_list
    bp4dP_all.all_aus = avail_label_list

    dataloaders_external = DataLoader(
        bp4dP_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    return dataloaders_train, dataloaders_test, dataloaders_external, avail_label_list


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
            return focal_loss.sum() / torch.sum(mask_1)
        elif self.reduce == "sum":
            return focal_loss.sum()
        elif self.reduce == "none":
            return focal_loss
        else:
            raise ValueError("not valid reduction method")


def train_data(configs, num_epoch=12, checkpoint_dir=None):
    """
    takes in a config file
    """
    tb = SummaryWriter(log_dir='/Storage/Projects/FER_GAN/code/local_global/train_stats/')

    fuse_weight1 = [configs['fusion_AU1']]+[configs['fusion_AU2']]+[configs['fusion_AU4']]+[0.5,0.5,0.5,0.5,0.5]\
                    +[configs['fusion_AU15']]+[0.5]+[configs['fusion_AU23']]+[configs['fusion_AU24']]
    # Parameters:
    fusion_weights = torch.Tensor(
        fuse_weight1
    )
    gamma_params = [configs['gamma_AU1']] + [configs['gamma_AU2']] + [configs['gamma_AU4']] + \
                    [0, 0, 0, 0, 0] + [configs['gamma_AU15']] + [0] + [configs['gamma_AU23']] + \
                        [configs['gamma_AU24']]

    model = localGlobalNet( 
                in_chs=3,
                num_classes=configs["n_aus"],
                layers=configs['layers'],
                num_cuts=configs["num_cuts"],
                channel_multiplier=1,
                block=configs['block'])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    fusion_weights = fusion_weights.to(device)

    if configs["loss"].lower() == "bce":
        criterion = normal_BCELoss(
            device=device, weights=configs["weights"], reduction='none'
        ).to(device)
        out_channel_multiplier = 1
    elif configs["loss"].lower() == "bce_prob":
        criterion = normal_BCELosswProb(device=device).to(device)
        out_channel_multiplier = 1
    elif configs["loss"].lower() == "multiclass_bce":
        criterion = custom_BCELoss(device=device).to(device)
        out_channel_multiplier = 2
    elif configs["loss"].lower() == "focal":
        criterion = multiclassFocalLoss(
            alpha=torch.Tensor(configs["weights"]),
            gamma=torch.Tensor(gamma_params),
            reduction="mean",
            device=device,
        ).to(device)
        out_channel_multiplier = 1

    criterion_dice = dice_loss(device=device, weight=configs['weights'], smooth=configs['smooth'])

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr = configs['lr'],
        momentum=configs['momentum'],
        weight_decay=configs['weight_decay'],
        nesterov=True
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=configs["step_size"],
            gamma=configs["lr_gamma"],
        )
        
    train_set, val_set, external_set, avail_label_list = load_data(sample_dat=configs['sample_dat'], batch_size=48, \
                                                                    train_mode='extreme', split="subject")

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        model.train()
        loss_train_total = 0
        running_loss = 0.0
        dic_items = list(train_set.items())
        random.shuffle(dic_items)

        for task, task_set in enumerate(dic_items):
            task_train_preds = {}
            task_train_labels = {}
            task_train_stats = collections.defaultdict(list)
            for batch_index, (_, img, label) in tqdm(enumerate(task_set[1])):
                img = Variable(img).to(device, dtype=torch.float)
                label = Variable(label).long().to(device)

                optimizer.zero_grad()
                pred1, pred2 = model(img)
                pred = (
                    fusion_weights.expand_as(pred1) * pred1
                    + (1 - fusion_weights).expand_as(pred2) * pred2
                )
                loss_ce = criterion(pred1, label)
                loss_dice = criterion_dice(nn.Sigmoid()(pred), label)
                loss = configs['beta1'] * loss_ce + (1-configs['beta1']) * loss_dice

                # Convert
                loss_train_total += loss.item()
                loss.backward()
                optimizer.step()

                if out_channel_multiplier == 1:
                    for aus in range(len(avail_label_list)):
                        au_realName = avail_label_list[
                            aus
                        ]  # Real Name for Action Units

                        pred_val = (
                            (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5)
                            .int()
                            .numpy())

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

        print(f'Loss: {loss_train_total}')
        print(f'accuracy: {np.mean([mm[2] for mm in list(task_train_stats.values())])}')

        tb.add_scalar('Loss/train', loss_train_total, epoch)
        tb.add_scalar('AVG F1/train', np.mean([mm[2] for mm in list(task_train_stats.values())]), epoch)

        model.eval()
        with torch.no_grad():
            loss_val_total = 0
            dic_items = list(val_set.items())
            random.shuffle(dic_items)

            for task, task_set in enumerate(dic_items):
                task_val_preds = {}
                task_val_labels = {}
                task_val_stats = collections.defaultdict(list)
                for batch_index, (_, img, label) in tqdm(enumerate(task_set[1])):
                    img = Variable(img).to(device, dtype=torch.float)
                    label = Variable(label).long().to(device)

                    pred1, pred2 = model(img)
                    pred = (
                        fusion_weights.expand_as(pred1) * pred1
                        + (1 - fusion_weights).expand_as(pred2) * pred2
                    )
                    loss_ce = criterion(pred, label) 
                    loss_dice = criterion_dice(nn.Sigmoid()(pred), label)
                    loss = configs['beta1'] * loss_ce + (1-configs['beta1']) * loss_dice
                    
                    loss_val_total += loss.item()

                    if out_channel_multiplier == 1:
                        for aus in range(len(avail_label_list)):
                            au_realName = avail_label_list[aus]
                            if label[0, aus] != -1:
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

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)
        
        if epoch >= 4:
            scheduler.step()

        print(f'Loss: {loss_val_total}')
        print(f'accuracy: {np.mean([mm[2] for mm in list(task_val_stats.values())])}')

        tb.add_scalar('Loss/val', loss_val_total, epoch)
        tb.add_scalar('AVG F1/val', np.mean([mm[2] for mm in list(task_val_stats.values())]), epoch)

        for au_label in avail_label_list:
            tb.add_scalar(f'F1 ACC {au_label}/val', task_val_stats[au_label][2], epoch)
        # tune.report(loss=loss_val_total, accuracy= np.mean([mm[2] for mm in list(task_val_stats.values())]))
    
    tb.close()

def test_best_model(best_trial, checkpoint_dir):
    
    fuse_weight1 = [best_trial.config['fusion_AU1']]+[best_trial.config['fusion_AU2']]+[best_trial.config['fusion_AU4']]+[0.5,0.5,0.5,0.5,0.5]\
                    +[best_trial.config['fusion_AU15']]+[0.5]+[best_trial.config['fusion_AU23']]+[best_trial.config['fusion_AU24']]
    # Parameters:
    fusion_weights = torch.Tensor(
        fuse_weight1
    )

    gamma_params = [best_trial.config['gamma_AU1']] + [best_trial.config['gamma_AU2']] + [best_trial.config['gamma_AU4']] + \
                    [0, 0, 0, 0, 0] + [best_trial.config['gamma_AU15']] + [0] + [best_trial.config['gamma_AU23']] + \
                        [best_trial.config['gamma_AU24']]

    best_trained_model = localGlobalNet( 
                in_chs=3,
                num_classes=best_trial.config["n_aus"],
                layers=best_trial.config['layers'],
                num_cuts=best_trial.config["num_cuts"],
                channel_multiplier=1,
                block=best_trial.config['block'])
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)
    fusion_weights = fusion_weights.to(device)

    if best_trial.config["loss"].lower() == "bce":
        criterion = normal_BCELoss(
            device=device, weights=best_trial.config["weights"], reduction='none'
        ).to(device)
        out_channel_multiplier = 1
    elif best_trial.config["loss"].lower() == "bce_prob":
        criterion = normal_BCELosswProb(device=device).to(device)
        out_channel_multiplier = 1
    elif best_trial.config["loss"].lower() == "multiclass_bce":
        criterion = custom_BCELoss(device=device).to(device)
        out_channel_multiplier = 2
    elif best_trial.config["loss"].lower() == "focal":
        criterion = multiclassFocalLoss(
            alpha=torch.Tensor(best_trial.config["weights"]),
            gamma=torch.Tensor(gamma_params),
            reduction="mean",
            device=device,
        ).to(device)
        out_channel_multiplier = 1

    criterion_dice = dice_loss(device=device, weight=best_trial.config['weights'], smooth=best_trial.config['smooth'])

    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        model_state, _ = torch.load(checkpoint_path)
        best_trained_model.load_state_dict(model_state)

    train_set, val_set, external_set, avail_label_list = load_data(sample_dat=best_trial.config['sample_dat'], batch_size=48, \
                                                                    train_mode='extreme', split="subject")

    with torch.no_grad():

        best_trained_model.eval()
        test_stats = collections.defaultdict(list)
        loss_test_total = 0
        test_preds = {}
        test_labels = {}

        with torch.no_grad():
            loss_test_total = 0
            test_preds = {}
            test_labels = {}
            for batch_index, (_, img, label) in tqdm(enumerate(external_set)):

                img = Variable(img).to(device, dtype=torch.float)
                label = Variable(label).long().to(device)

                pred1, pred2 = best_trained_model(img)
                pred = (
                    fusion_weights.expand_as(pred1) * pred1
                    + (1 - fusion_weights).expand_as(pred2) * pred2
                )
                loss_ce = criterion(pred, label)
                loss_dice = criterion_dice(nn.Sigmoid()(pred), label)
                loss = best_trial.config['beta1'] * loss_ce + (1-best_trial.config['beta1']) * loss_dice
                loss = loss.sum()
                loss_test_total += loss.item()

                recorded_AU = []
                recorded_f1 = []
                recorded_acc = []
                for aus in range(len(avail_label_list)):
                    au_realName = avail_label_list[aus]
                    if label[0, aus] != -1:
                        if out_channel_multiplier == 1:
                            tmp_pred = (
                                (F.sigmoid(pred.detach().cpu())[:, aus] > 0.5)
                                .int()
                                .numpy()
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

    for aus in test_preds.keys():
        print(f"aus:{aus}")
        au_label_vals = test_labels[aus]
        au_pred_vals = test_preds[aus]
        au_label_valid_idx = np.where(
            np.logical_or(au_label_vals == 0, au_label_vals == 1)
        )

        thresholded_pred = np.array([int(i > 0.5) for i in au_pred_vals])
        (
            au_spec_prec,
            au_spec_rec,
            au_spec_fsco,
            au_spec_supp,
        ) = precision_recall_fscore_support(
            thresholded_pred[au_label_valid_idx],
            au_label_vals[au_label_valid_idx],
            pos_label=1,
            average="binary",
        )
        print(au_spec_fsco)


    au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp = precision_recall_fscore_support(au_pred_vals[au_label_valid_idx], au_label_vals[au_label_valid_idx], pos_label=1, average='binary')
    test_stats[aus] = [au_spec_prec, au_spec_rec, au_spec_fsco, au_spec_supp]
    print(f'avg test f1 is:{np.mean([mm[2] for mm in list(test_stats.values())])}')


def main(num_samples=36, max_num_epochs=16):

    configs = {
        "n_aus": 12,
        "batch_size": 48,
        "model_type": "localglobal",
        "extreme_transform": "extreme",
        "data_split_method": "subject",
        "loss": "focal",
        "sample_dat": 0.5,

        'layers': tune.choice([[1, 1, 2, 2], [2,2,2,2]]),
        "num_cuts": tune.choice([2,4]),
        'block': tune.choice(['basicblock', 'bottleneck']),

        'gamma_AU1': tune.choice([0, 0.5, 1, 2]), 
        'gamma_AU2': tune.choice([0, 0.5, 1, 2]), 
        'gamma_AU4': tune.choice([0, 0.5, 1, 2]), 
        'gamma_AU15': tune.choice([0, 0.5, 1, 2]), 
        'gamma_AU23': tune.choice([0, 0.5, 1, 2]), 
        'gamma_AU24': tune.choice([0, 0.5, 1, 2]), 
        'fusion_AU1': tune.choice([0.2, 0.5, 0.8]), 
        'fusion_AU2': tune.choice([0.2, 0.5, 0.8]),
        'fusion_AU4': tune.choice([0.2, 0.5, 0.8]),
        'fusion_AU15': tune.choice([0.2, 0.5, 0.8]), 
        'fusion_AU23': tune.choice([0.2, 0.5, 0.8]), 
        'fusion_AU24': tune.choice([0.2, 0.5, 0.8]), 
        'fusion_weight': [0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3],
        'beta1':tune.uniform(0.1, 0.9),
        
        "lr": tune.loguniform(0.0001, 0.01),
        "momentum": tune.uniform(0.1, 0.9),
        'weight_decay': tune.choice([1e-5, 1e-4]),

        "weights": [4.45,5.65,5.27,2.18,1.89,1.70,1.77,2.23,6.48,2.92,5.97,6.97],
        'smooth': tune.uniform(0.1, 1),

        'step_size': tune.choice([2,4]),
        'lr_gamma': tune.choice([0.1, 0.5])
    }


    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=4,
        reduction_factor=2)

    result = tune.run(
        tune.with_parameters(train_data, num_epoch=max_num_epochs),
        resources_per_trial={"cpu": 8, "gpu": 1/3},
        config=configs,
        metric="accuracy",
        mode="max",
        num_samples=num_samples,
        scheduler=scheduler,
        local_dir="/Storage/Projects/tune_results",
        name='tune_result1'
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    test_best_model(best_trial)


class wrap_class(nn.Module):

    def __init__(self, config) -> None:
        super(wrap_class, self).__init__()
        self.config = config

if __name__ == "__main__":

    configs = {
        "n_aus": 12,
        "batch_size": 48,
        "model_type": "localglobal",
        "extreme_transform": "extreme",
        "data_split_method": "subject",
        "loss": "focal",
        "sample_dat": 0.5,

        'layers': [1, 1, 2, 2],
        "num_cuts": 2,
        'block': 'basicblock',

        'gamma_AU1': 0, 
        'gamma_AU2': 0, 
        'gamma_AU4': 0, 
        'gamma_AU15': 0, 
        'gamma_AU23': 0, 
        'gamma_AU24': 0, 
        'fusion_AU1': 0.2,
        'fusion_AU2': 0.2, 
        'fusion_AU4': 0.2, 
        'fusion_AU15': 0.2, 
        'fusion_AU23': 0.2, 
        'fusion_AU24': 0.2,  
        'fusion_weight': [0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3],
        'beta1':0.5,
        
        "lr": 1e-2,
        "momentum": 0.9,
        'weight_decay': 1e-5, 

        "weights": [4.45,5.65,5.27,2.18,1.89,1.70,1.77,2.23,6.48,2.92,5.97,6.97],
        'smooth': 1,

        'step_size': 4,
        'lr_gamma': 0.5
    }
    
    train_data(configs, num_epoch=16, checkpoint_dir=None)

    # configs = {
    #     'layers':[1, 1, 2, 2],
    #     'gamma_P1': [0.5,0.5,0.5], 
    #     'gamma_P2': [0.5,0.5,0.5,0.5], 
    #     'fusion_AU1': 0.5, 
    #     'fusion_AU2': 0.5,
    #     'fusion_AU4': 0.5,
    #     'fusion_AU15': 0.5,
    #     'fusion_AU23': 0.5, 
    #     'fusion_AU24': 0.5, 
    #     'fusion_weight': [0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3],
    #     'beta1':0.5,
    #     "n_aus": 12,
    #     "batch_size": 64,
    #     "lr": 1e-4,
    #     "model_type": "localglobal",
    #     "extreme_transform": "extreme",
    #     "weights": [
    #         7.9420,
    #         9.0383,
    #         11.4478,
    #         1.9043,
    #         1.4496,
    #         1.4818,
    #         1.6515,
    #         1.5934,
    #         7.3480,
    #         6.2880,
    #         5.1076,
    #         14.6253,
    #     ],
    #     "data_split_method": "subject",
    #     "loss": "focal",
    #     "weight_decay": 1e-5,
    #     "sample_dat": 0.5,
    #     "beta": 0.5,
    #     "num_cuts": 4,
    # }

    # config1 = {"batch_size": 64,
    #             "beta": 0.5,
    #             "beta1": 0.2,
    #             "data_split_method": "subject",
    #             "extreme_transform": "extreme",
    #             "fusion_AU1": 0.8,
    #             "fusion_AU15": 0.5,
    #             "fusion_AU2": 0.2,
    #             "fusion_AU23": 0.5,
    #             "fusion_AU24": 0.5,
    #             "fusion_AU4": 0.5,
    #             "fusion_weight": [0.5,0.5,0.7,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.3],
    #             "gamma_P1": [1,1,1],
    #             "gamma_P2": [0.5,0.5,0.5,0.5],
    #             "layers": [1,1,2,2],
    #             "loss": "focal",
    #             "lr": 3e-05,
    #             "model_type": "localglobal",
    #             "n_aus": 12,
    #             "num_cuts": 2,
    #             "sample_dat": None,
    #             "weight_decay": 1e-05,
    #             "weights": [
    #                 7.942,
    #                 9.0383,
    #                 11.4478,
    #                 1.9043,
    #                 1.4496,
    #                 1.4818,
    #                 1.6515,
    #                 1.5934,
    #                 7.348,
    #                 6.288,
    #                 5.1076,
    #                 14.6253]
    #             }

    # best_model = wrap_class(config=config1)
    # test_best_model(best_trial=best_model,checkpoint_dir='/root/ray_results/train_data_2022-05-25_17-04-26/train_data_42e61_00012_12_beta1=0.2,fusion_AU1=0.8,fusion_AU15=0.5,fusion_AU2=0.2,fusion_AU23=0.5,fusion_AU24=0.5,fusion_AU4=0.5,g_2022-05-25_19-51-38/checkpoint_000006')
    # # train_data(configs)
    # main(num_samples=32, max_num_epochs=16)
