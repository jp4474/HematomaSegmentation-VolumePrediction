# HematomaSegmentation-VolumePrediction

## Preprocessing
Run the pre-processing script to convert the dataset (nii.gz files and npy files) to npz format:

```bash
python3 pre_CT_MR.py \
    -img_path data/img_train_val\
    -img_name_suffix .nii.gz \
    -gt_path data/mask_train_val\
    -gt_name_suffix .npy \
    -output_path 230620_6L_64C_GN/data \
    -num_workers 4 \
    -modality CT \
    -anatomy Brain \
    -window_level 65 \
    -window_width 70 \
    --save_nii
```
Convert npz to npy

```bash
python3 npz_to_npy.py \
    -npz_dir data/MedSAM_val/CT_Brain \
    -npy_dir data/npy_val \
    -num_workers 4
```
