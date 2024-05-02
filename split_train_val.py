# write a script to split the training data into training and validation data with 0.9/0.1 ratio.

from sklearn.model_selection import train_test_split
import glob
from pathlib import Path


if __name__ == "__main__":
    img_train_val_test = Path(r'data/img_train_val') # WindowsPath of mri scans folder
    mask_train_val_pathes = sorted(Path(r"data/mask_train_val").glob('*.npy')) # WindowsPath of all masks
    img_train_val_pathes = [img_train_val_test / p.name.replace('.npy', '.nii.gz') for p in mask_train_val_pathes] # WindowsPath of all mri scans using the same order as the masks
    print(f'len(img_train_val_pathes) = {len(img_train_val_pathes)}')
    print(f'len(mask_train_val_pathes) = {len(mask_train_val_pathes)}')
    train_size = 0.9
    random_seed = 230620
    img_train_pathes, img_val_pathes, mask_train_pathes, mask_val_pathes = train_test_split(
        img_train_val_pathes, mask_train_val_pathes, train_size=train_size, random_state=random_seed
    )
    print(f'len(img_train_pathes) = {len(img_train_pathes)}')
    print(f'len(img_val_pathes) = {len(img_val_pathes)}')
    print(f'len(mask_train_pathes) = {len(mask_train_pathes)}')
    print(f'len(mask_val_pathes) = {len(mask_val_pathes)}')

    with open('data/img_train_pathes.txt', 'w') as f:
        for path in img_train_pathes:
            path = str(path).split('/')[-1]
            f.write(path + '\n')

    with open('data/mask_train_pathes.txt', 'w') as f:
        for path in mask_train_pathes:
            path = str(path).split('/')[-1]
            f.write(path + '\n')
    with open('data/img_val_pathes.txt', 'w') as f:
        for path in img_val_pathes:
            path = str(path).split('/')[-1]
            f.write(path + '\n')

    with open('data/mask_val_pathes.txt', 'w') as f:
        for path in mask_val_pathes:
            path = str(path).split('/')[-1]
            f.write(path + '\n')
