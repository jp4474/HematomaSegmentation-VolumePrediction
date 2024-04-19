import os
import shutil
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import nibabel as nib
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torchmetrics.classification import BinaryJaccardIndex, Dice, JaccardIndex
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.nn.modules.loss import BCEWithLogitsLoss
from monai.losses.dice import *


from monai.data import DataLoader 
import pytorch_lightning as pl
import lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class BraTSDataset(Dataset):    
    def __init__(self, data_root_folder, folder = '', n_sample=None):
        main_folder = os.path.join(data_root_folder, folder)
        self.folder_path = os.path.join(main_folder, 'slice')
        #self.file_names = sorted(os.listdir(self.folder_path))[:n_sample]


    def __getitem__(self, index):
        file_name = os.listdir(self.folder_path)[index]
        #file_name = self.file_names[index]
        sample = np.load(os.path.join(self.folder_path, file_name))
        #eps = 0.0001
        img = sample[0,:,:]
        diff = np.subtract(img.max(), img.min(), dtype=np.float64)
        denom = np.clip(diff, a_min=1e-8, a_max=None)
        img = (img - img.min()) / denom
        mask = sample[1, :, :]
        mask[mask>0.0] = 1.0
        mask[mask==0.0] = 0
        img_as_tensor = np.expand_dims(img, axis=0)
        mask_as_tensor = np.expand_dims(mask, axis=0)
        
        
        return {
            'image': img_as_tensor,
            'mask': mask_as_tensor,
            'img_id': file_name
        }
 
    def __len__(self):
        return len(os.listdir(self.folder_path))
        #return len(self.file_names)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class resconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)

        return residual + x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(32, ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, first_layer_numKernel=64, name = "U_Net"):
        super(U_Net, self).__init__()
        self.name = name
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=first_layer_numKernel)
        self.Conv2 = conv_block(ch_in=first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Conv3 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Conv4 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Conv5 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=16 * first_layer_numKernel)

        self.Up5 = up_conv(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)
        self.Up_conv5 = conv_block(ch_in=16 * first_layer_numKernel, ch_out=8 * first_layer_numKernel)

        self.Up4 = up_conv(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)
        self.Up_conv4 = conv_block(ch_in=8 * first_layer_numKernel, ch_out=4 * first_layer_numKernel)

        self.Up3 = up_conv(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)
        self.Up_conv3 = conv_block(ch_in=4 * first_layer_numKernel, ch_out=2 * first_layer_numKernel)

        self.Up2 = up_conv(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)
        self.Up_conv2 = conv_block(ch_in=2 * first_layer_numKernel, ch_out=first_layer_numKernel)

        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(first_layer_numKernel, output_ch, kernel_size=1, stride=1, padding=0) # Use sigmoid activation for binary segmentation
        )
        # self.Conv_1x1 =  nn.Conv2d(first_layer_numKernel, output_ch, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        return d1
    
def dice_coeff_binary(y_pred, y_true):
        """Values must be only zero or one."""
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        eps = 0.0001
        inter = torch.dot(y_pred.view(-1).float(), y_true.view(-1).float())
        union = torch.sum(y_pred.float()) + torch.sum(y_true.float())
        return ((2 * inter.float() + eps) / (union.float() + eps))


class U_Net_DDP(pl.LightningModule):
    def __init__(self, net, lr, loss, jaccard):
        super().__init__()
        self.net = net
        self.lr = lr
        self.loss = loss 
        #self.dice = dice
        self.jaccard = jaccard
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        imgs = batch['image'].float()
        true_masks = batch['mask']

        y_pred = self(imgs)
        loss = self.loss(y_pred, true_masks.float())
        #y_pred = (y_pred >= 0.5).float()
        y_pred = self.sigmoid(y_pred)
        

        batch_dice_score = dice_coeff_binary(y_pred, true_masks)
        batch_jaccard_score = jaccard_index_metric(y_pred, true_masks)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("dice", batch_dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("jaccard", batch_jaccard_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs = batch['image'].float()
        true_masks = batch['mask']
        
        y_pred = self(imgs)
        loss = self.loss(y_pred, true_masks.float())
        #y_pred = (y_pred >= 0.5).float()
        y_pred = self.sigmoid(y_pred)

        batch_dice_score = dice_coeff_binary(y_pred, true_masks)
        batch_jaccard_score = jaccard_index_metric(y_pred, true_masks)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_dice", batch_dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_jaccard", batch_dice_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs * 200, 0)
        return [optimizer] #, [scheduler]

if __name__ == '__main__':
    data_root_folder = '/home/jupyter/full_raw - Copy'
    train_dataset = BraTSDataset(data_root_folder = data_root_folder, folder = 'train')
    val_dataset = BraTSDataset(data_root_folder = data_root_folder, folder = 'val')
    test_dataset = BraTSDataset(data_root_folder = data_root_folder, folder = 'test')
    BATCH_SIZE = 16
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    validation_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    checkpointing = ModelCheckpoint(monitor="val_loss",
                                dirpath='/home/jupyter/checkpoints/',
                                filename='unet-epoch-{epoch}-{val_loss:.2f}-{val_dice:.2f}-{val_jaccard:.2f}', 
                                save_top_k=-1)
    es = EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(precision=16, 
                        devices=2, 
                        accelerator="gpu",
                        strategy="ddp_notebook", 
                        max_epochs=10, 
                        callbacks=[es, checkpointing])
    net = U_Net(img_ch=1, output_ch=1)
    lr = 1e-3
    loss = GeneralizedDiceFocalLoss()
    jaccard_index_metric = BinaryJaccardIndex()
    model = U_Net_DDP(net, lr, loss, jaccard_index_metric)
    trainer.fit(model, train_dataloader, validation_dataloader)
