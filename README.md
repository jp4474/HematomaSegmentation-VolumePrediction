# HematomaSegmentation-VolumePrediction

## File Description


## Folder Structure 
Folder structure for data
```bash
└── data
    ├── divide.sh
    ├── img_test
    ├── img_train
    ├── img_train_pathes.txt // file names for img train
    ├── img_val
    ├── img_val_pathes.txt // file names for img val
    ├── mask_test
    ├── mask_train
    ├── mask_train_pathes.txt // file names for mask train
    ├── mask_val
    ├── mask_val_pathes.txt // file names for mask val
    ├── train_final_npy
    │   ├── gts
    │   └── imgs
    ├── val_final_npy
    │   ├── gts
    │   └── imgs
    └──train_final_npy
        ├── gts
        └── imgs
```
## Preprocessing
Run the pre-processing script to convert the dataset (nii.gz files and npy files) to npz format:

```bash
python pre_CT_MR.py `
    -img_path img_train `
    -img_name_suffix .nii.gz `
    -gt_path mask_train `
    -gt_name_suffix .npy ` // expects npy extension but the script can be modified to take in .nii.gz
    -output_path train_final_npz `
    -num_workers 4 `
    -modality CT ` // the script is only viable for CT modality. Modify the if-else statement in the script for other modalities 
    -anatomy Brain ` 
    -window_level 65 `
    -window_width 70 `
    --save_nii
```
Convert npz to npy

```bash
python npz_to_npy.py `
    -npz_dir train_final_npz `
    -npy_dir train_final_npy `
    -num_workers 4

```
1. Run `divide.sh` to split img_train_val_test folder data into img_train_val and img_test
2. Use `split_train_val.py` to generate text files containing file names for img_train, mask_train, img_val, and mask_val
3. Run `split.sh` to split the files based on the text files. i.e. train file should be in img_train. val file should be in img_val
4. Run pre_CT_MR to preprocess CT scans
5. Convert npz to npy using npz_to_npy
6. Load the dataset using npyDataset Class in 'medsam_train_native.py'
7. Start training by (on a single GPU)
   ```bash
    python medsam_train_native.py
   ```

## Acknowledgements
- We thank the Columbia University's Biomedical Engineering Department, the teaching team of BMENE4460, and Columbia University Irving Medical Center for providing the dataset.
  - Special Thanks to Dr. Jia Guo for advising our team.
- We thank Meta AI and BoWangLab for making the source code publicly available.

## References

```bash
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  pages={1--9},
  year={2024}
}
```
