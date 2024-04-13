import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
import torch


def compute_volume_from_raw_mask(data_root_folder, folder = ''):
    main_folder = os.path.join(data_root_folder, folder)
    masks_dir = os.path.join(main_folder, 'mask')
    
    masks_file_name = sorted(glob.glob(os.path.join(masks_dir, '*.nii.gz')))

    file_names = []
    volumes = []

    file_id =  list(file.split("_")[1] for file in os.listdir(masks_dir))
    for (file_id,  mask_file_name) in zip(file_id,  masks_file_name):
        single_mask_path = os.path.join(masks_dir, mask_file_name)
        single_mask_nii = nib.load(single_mask_path)
        single_mask_array = single_mask_nii.get_fdata()
        single_mask_array = single_mask_array.astype(np.int16)
        single_mask_array = np.expand_dims(single_mask_array, axis=0)
        single_mask_array = single_mask_array.astype(np.int16)
        # check: xy should be 240x240
        #print(single_mask_array.shape)
        assert single_mask_array.shape[1]==240, 'Dimension mismatch'
        assert single_mask_array.shape[2]==240, 'Dimension mismatch'

        single_mask_array[single_mask_array > 0.0] = 1.0
        single_mask_array[single_mask_array == 0.0] = 0.0

        volume = np.sum(single_mask_array)
        volumes.append(volume)
        file_names.append(file_id)

    df = pd.DataFrame({'file_name': file_names, 'volume': volumes})
    df.to_csv(os.path.join(main_folder, f'{folder}_volume.csv'))

def compute_volume_from_prediction(torch_mat_binary):
    return torch.sum(torch_mat_binary) # torch mat is a binary matrix



if __name__ == "__main__":
    folder = r"C:\Users\lisag\OneDrive\Bureau\Columbia_Coursework\Spring_2023\DLBI\data_project\full_raw"
    compute_volume_from_raw_mask(data_root_folder = folder, folder='train')
    compute_volume_from_raw_mask(data_root_folder = folder, folder='val')
    compute_volume_from_raw_mask(data_root_folder = folder, folder='test')
        

       