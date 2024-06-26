# HematomaSegmentation-VolumePrediction

## File Description

EDA folder <br />
    `ICH_dataset.ipynb`: ICH dataset exploration and visualization <br />
<br />
data folder <br />
    `divide.sh`: script to move test images from img_train_val_test folder to img_test folder <br />
    `split_train_val.py`: script to split the training data into training set and validation set with 0.9/0.1 ratio (creates .txt files with scans ids) <br />
    `split.sh`: script to move train (val) images/masks from img_train_val/mask_train_val folder to img_train/mask_train (img_val/mask_val) folder <br />
    `pre_CT_MR.py`: script to preprocess data <br />
    `npz_to_npy.py`: script to convert .npz files outputed by `pre_CT_MR.py` to .npy files <br />
<br />
segment anything folder <br />
    - folder contains MedSAM model <br />
<br />
`lite_medsam.pth`: weights of not fine-tuned LiteMedSAM model (weights of fine-tuned model are available upon request) <br />
`lora_train.py`: script for fine-tuning LiteMedSAM with LoRA <br />
`tiny_vit_sam.py`: script containing the class for LiteMedSAM <br />
`visualize_pred.py`: script for visualizing the segmentation results across models <br />
`inference.py`: script for inference on all models <br />

## Folder Structure 
Folder structure for data
```bash
└── data
    ├── divide.sh
    ├── npz_to_npy.py
    ├── pre_CT_MR.py
    ├── split_train_val.py
    ├── split.sh
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
    └── train_final_npy
        ├── gts
        └── imgs
```
## Preprocessing
1. Run `divide.sh` to split img_train_val_test folder data into img_train_val and img_test
2. Use `split_train_val.py` to generate text files containing file names for img_train, mask_train, img_val, and mask_val
3. Run `split.sh` to split the files based on the text files. i.e. train file should be in img_train. val file should be in img_val
4. Run pre_CT_MR to preprocess CT scans
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
6. Convert npz to npy using npz_to_npy
```bash
  python npz_to_npy.py `
  -npz_dir train_final_npz `
  -npy_dir train_final_npy `
  -num_workers 4
```
7. Load the dataset using npyDataset Class in 'lora_train.py'
8. Start training by (on a single GPU)
```bash
  python lora_train.py \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --rank 32 \
  --alpha 64 \
  --dropout 0.1 \
  --epochs 10 \
  --use_rlora True \
  --eval_steps 10 \
  --gradient_accumulation_steps 8
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
