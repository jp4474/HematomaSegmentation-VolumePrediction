# HematomaSegmentation-VolumePrediction

## Preprocessing
Run the pre-processing script to convert the dataset (nii.gz files and npy files) to npz format:

```bash
python pre_CT_MR.py `
    -img_path img_train `
    -img_name_suffix .nii.gz `
    -gt_path mask_train `
    -gt_name_suffix .npy `
    -output_path train_final_npz `
    -num_workers 4 `
    -modality CT `
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
