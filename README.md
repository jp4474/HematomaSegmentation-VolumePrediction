# HematomaSegmentation-VolumePrediction

## Preprocessing
Run the pre-processing script to convert the dataset (nii.gz files and npy files) to npz format:

```bash
python3 pre_CT_MR.py \
    -img_path 230620_6L_64C_GN/data/img_train_val_test\
    -img_name_suffix .nii.gz \
    -gt_path 230620_6L_64C_GN/data/mask_train_val_test\
    -gt_name_suffix .npy \
    -output_path 230620_6L_64C_GN/data \
    -num_workers 4 \
    -modality CT \
    -anatomy Brain \
    -window_level 40 \
    -window_width 400 \
    --save_nii
```
Convert npz to npy

```bash
python3 npz_to_npy.py \
    -npz_dir data/MedSAM_val/CT_Brain \
    -npy_dir data/npy_val \
    -num_workers 4
```
# TODO:
- find window_level, window_width
- modify npydataset class
