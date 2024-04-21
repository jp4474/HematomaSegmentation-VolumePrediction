import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import SamModel, SamProcessor
import torch.nn.functional as F
import os
from torchmetrics.classification import BinaryJaccardIndex, Dice, JaccardIndex
from transformers import AutoConfig
from peft import LoftQConfig, LoraConfig, get_peft_model

def dice_coeff_binary(y_pred, y_true):
    """Values must be only zero or one."""
    eps = 0.0001
    inter = torch.dot(y_pred.view(-1).float(), y_true.view(-1).float())
    union = y_pred.float().sum() + y_true.float().sum()
    return ((2 * inter.float() + eps) / (union.float() + eps))


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

class BraTSDataset(Dataset):    
    def __init__(self, data_root_folder, folder = '', n_sample=None):
        main_folder = os.path.join(data_root_folder, folder)
        self.folder_path = os.path.join(main_folder, 'slice')
        self.file_names = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        #self.file_names = sorted(os.listdir(self.folder_path))[:n_sample]


    def __getitem__(self, index):
        file_name = self.file_names[index]
        sample = np.load(os.path.join(self.folder_path, file_name), allow_pickle=True)
        #eps = 0.0001
        img = sample[0,:,:]
        #img = img.resize((256, 256)) 
        diff = np.subtract(img.max(), img.min(), dtype=np.float64)
        denom = np.clip(diff, a_min=1e-8, a_max=None)
        img = (img - img.min()) / denom
        mask = sample[1, :, :]
        #mask= mask.resize((256, 256)) 
        mask[mask > 0] = 1
        mask[mask == 0] = 0
        
        gt2D = np.uint8(
            mask == 1
        )

        y_indices, x_indices = np.where(gt2D > 0)  #
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min) #- random.randint(0, self.bbox_shift))
        x_max = min(W, x_max) #+ random.randint(0, self.bbox_shift))
        y_min = max(0, y_min) #- random.randint(0, self.bbox_shift))
        y_max = min(H, y_max) #+ random.randint(0, self.bbox_shift))
        
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        img_as_tensor = np.expand_dims(img, axis=0)
        img_as_tensor = np.repeat(img_as_tensor, 3, axis=0)

        mask_as_tensor = np.expand_dims(mask, axis=0)
        bboxes_as_tensor = np.expand_dims(bboxes, axis=0)

        img_as_tensor = torch.from_numpy(img_as_tensor)
        mask_as_tensor = torch.from_numpy(mask_as_tensor)
        bboxes_as_tensor = torch.from_numpy(bboxes_as_tensor)
        
        #return img_as_tensor, mask_as_tensor
        return {
            'image': img_as_tensor.to(dtype=torch.float32),
            'mask': mask_as_tensor.to(dtype=torch.int64),
            'box' : bboxes_as_tensor.to(dtype=torch.float32),
            'img_id': file_name
        }
        
    def __len__(self):
        return len(self.file_names)
    

if __name__ == "__main__":
    # img = np.random.rand(256, 256)
    # mask = np.random.rand(256, 256)
    # pred = np.random.rand(256, 256)
    # visualize_pred(img, mask, pred)

    # model = MyModel()

    # # Load the state dictionary
    # model.load_state_dict(torch.load('path_to_your_model.pth'))
    
    # model = torch.load("/Users/jun/Desktop/Personal/BMENE/checkpoints/medsam_epoch_009.pth")
    # model.eval()

    processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base") #.to(device)
    config = AutoConfig.from_pretrained("wanglab/medsam-vit-base")
    model = SamModel(config)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="lora_only",
        target_modules=["qkv", "q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)

    model.load_state_dict(torch.load("/Users/jun/Desktop/BMENE/checkpoints/medsam_epoch_009.pth"))
    model.eval()

    data_root_folder = '/Users/jun/Desktop/BMENE/medsam_local/'
    dataset = BraTSDataset(data_root_folder, 'test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, data in enumerate(dataloader):
        img = data['image']
        mask = data['mask']
        box = data['box']

        with torch.no_grad():
            inputs = processor(img, input_boxes=box, do_normalize = False, do_rescale = False, return_tensors="pt")
            outputs = model(**inputs, multimask_output=False)        
            
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # convert soft mask to hard mask
            medsam_seg_prob = F.interpolate(
                medsam_seg_prob,
                size=(240, 240),
                mode="bilinear",
                align_corners=False,
            )

            medsam_seg = medsam_seg_prob > 0.5


            visualize_pred(img.squeeze(0).permute(1,2, 0), mask[0][0], medsam_seg[0][0])