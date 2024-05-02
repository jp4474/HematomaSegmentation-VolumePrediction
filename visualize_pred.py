import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import SamModel, SamProcessor
import torch.nn.functional as F
import os
import math
import nibabel as nib
from torchmetrics.classification import BinaryJaccardIndex, Dice, JaccardIndex
from transformers import AutoConfig
from peft import LoftQConfig, LoraConfig, get_peft_model
from medsam_train_native import npyDataset

# Function to calculate Dice coefficient
def dice_coeff_binary(y_pred, y_true):
    """Values must be only zero or one."""
    eps = 0.0001
    inter = torch.dot(y_pred.view(-1).float(), y_true.view(-1).float())
    union = y_pred.float().sum() + y_true.float().sum()
    return ((2 * inter.float() + eps) / (union.float() + eps))

# Function to visualize the image, mask, and prediction
def visualize_pred(img, mask, pred):
    # Flatten the mask and prediction arrays for score calculations
    dice = Dice()
    jaccard = BinaryJaccardIndex()
    # Calculate Dice and Jaccard scores
    dice_score = dice_coeff_binary(pred, mask)
    jaccard_score = jaccard(pred, mask)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(mask)
    ax[1].set_title(f"Ground Truth")
    ax[2].imshow(pred)
    ax[2].set_title(f"Prediction\nDice: {dice_score:.2f} Jaccard: {jaccard_score:.2f}")
    plt.show()
    

if __name__ == "__main__":
    # Load the image
    img = nib.load("/Users/lakon/Desktop/Columbia/HematomaSegmentation-VolumePrediction/data/img_test/38_2.nii.gz")

    # Convert the image data to numpy array
    img_data = img.get_fdata()

    # Find the first and last non-empty slices
    first_slice = 0
    last_slice = img_data.shape[2] - 1

    while np.all(img_data[:, :, first_slice] == 0):
        first_slice += 1

    while np.all(img_data[:, :, last_slice] == 0):
        last_slice -= 1

    # Calculate the number of plot rows and columns
    num_slices = last_slice - first_slice + 1
    num_cols = round(math.sqrt(num_slices))
    num_rows = num_cols if num_slices % num_cols == 0 else num_cols + 1

    # Create a figure for the plots
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Plot all slices
    for i in range(first_slice, last_slice + 1):
        row = (i - first_slice) // num_cols
        col = (i - first_slice) % num_cols
        ax[row, col].imshow(np.rot90(img_data[:, :, i]), cmap="gray")
        ax[row, col].set_title(f"Slice {i}")

    # Remove empty subplots
    if num_slices % num_cols != 0:
        for j in range(num_slices, num_rows * num_cols):
            fig.delaxes(ax.flatten()[j])

    plt.tight_layout()
    plt.show()