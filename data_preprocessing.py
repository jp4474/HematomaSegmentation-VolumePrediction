import os
import shutil
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import nibabel as nib



def genereate_npy(data_root_folder, folder = ''):
    folder = os.path.join(data_root_folder, folder)
    imgs_dir = os.path.join(folder, 'mri')
    masks_dir = os.path.join(folder, 'mask')
    
    imgs_file_name = sorted(glob.glob(os.path.join(imgs_dir, '*.nii.gz')))
    masks_file_name = sorted(glob.glob(os.path.join(masks_dir, '*.nii.gz')))

    assert len(imgs_file_name) == len(masks_file_name), 'There are some missing images or masks in {0}'.format(folder)

    file_id =  list(file.split("_")[1] for file in os.listdir(imgs_dir))
    for (file_id, img_file_name, mask_file_name) in zip(file_id, imgs_file_name, masks_file_name):
        single_image_path = os.path.join(imgs_dir, img_file_name)
        single_image_nii = nib.load(single_image_path)
        single_image_array = single_image_nii.get_fdata()
        single_image_array = single_image_array.astype(np.int16)
        single_image_array = np.expand_dims(single_image_array, axis=0)
        single_image_array = single_image_array.astype(np.int16)
        
        single_mask_path = os.path.join(masks_dir, mask_file_name)
        single_mask_nii = nib.load(single_mask_path)
        single_mask_array = single_mask_nii.get_fdata()
        single_mask_array = single_mask_array.astype(np.int16)
        single_mask_array = np.expand_dims(single_mask_array, axis=0)
        single_mask_array = single_mask_array.astype(np.int16)
        
        # get the last dimension size
        last_dim_size = single_image_array.shape[-1]
        
        assert last_dim_size==155, "Dimension mismatch"

        # iterate over the last dimension
        for i in range(last_dim_size):
            # get the slice
            
            img_slice_array = single_image_array[..., i]
            mask_slice_array = single_mask_array[..., i]
            img_slice_array = cv2.normalize(img_slice_array, None, 0, 1, cv2.NORM_MINMAX)
            mask_slice_array[mask_slice_array > 0.0] = 1.0
            mask_slice_array[mask_slice_array == 0.0] = 0.0
            stacked_array = np.dstack((img_slice_array, mask_slice_array))
            # save the slice as a .npy file
            np.save(os.path.join(folder, 'slice', f'{file_id}_slice_{i}.npy'), stacked_array)


if __name__ == "__main__":
    folder = r"C:\Users\lisag\OneDrive\Bureau\Columbia_Coursework\Spring_2023\DLBI\data_project\full_raw"
    genereate_npy(data_root_folder = folder, folder='train')
    genereate_npy(data_root_folder = folder, folder='val')
    genereate_npy(data_root_folder = folder, folder='test')