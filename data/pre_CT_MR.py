# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

import torch
import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from tqdm import tqdm
import cc3d

import multiprocessing as mp
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality", type=str, default="CT", help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="Abd",
                    help="Anaotmy name, [default: Abd]")
parser.add_argument("-img_name_suffix", type=str, default="_0000.nii.gz",
                    help="Suffix of the image name, [default: _0000.nii.gz]")
parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz",
                    help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-img_path", type=str, default="FLARE22Train/images",
                    help="Path to the nii images, [default: FLARE22Train/images]")
parser.add_argument("-gt_path", type=str, default="FLARE22Train/labels",
                    help="Path to the ground truth, [default: FLARE22Train/labels]")
parser.add_argument("-output_path", type=str, default="npz",
                    help="Path to save the npy files, [default: ./npz]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")
parser.add_argument("--save_nii", action="store_true",
                    help="Save the image and ground truth as nii files for sanity check; they can be removed")

args = parser.parse_args()

# convert nii image to npz files, including original image and corresponding masks
modality = args.modality  # CT or MR
anatomy = args.anatomy  # anantomy + dataset name
img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
prefix = modality + "_" + anatomy + "_"

nii_path = args.img_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
output_path = args.output_path  # path to save the preprocessed files
npz_tr_path = join(output_path, "MedSAM_train", prefix[:-1])
os.makedirs(npz_tr_path, exist_ok=True)

num_workers = args.num_workers

voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

save_nii = args.save_nii
# %% save preprocessed images and masks as npz files
def preprocess(name, npz_path):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npz_path : str
        path to save the npz files
    """
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_np = np.load(join(gt_path, gt_name))
    gt_np = np.transpose(gt_np, (2, 0, 1))  
    gt_data_ori = gt_np

    # remove small objects with less than 100 pixels in 2D slices

    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: for such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )

    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2 # 65, 70
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)  # transfroms np.min = lowerbound
            image_data_pre = (image_data_pre - lower_bound) / (upper_bound - lower_bound)
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = image_data
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(join(npz_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())



if __name__ == "__main__":
    tr_names = names

    preprocess_tr = partial(preprocess, npz_path=npz_tr_path)

    with mp.Pool(num_workers) as p:
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()

