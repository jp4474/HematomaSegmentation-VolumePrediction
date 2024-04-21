import os
from monai.losses import GeneralizedDiceFocalLoss
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import BinaryJaccardIndex, Dice
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
# from segment_anything import sam_model_registry
from transformers import SamModel, SamProcessor

class BraTSDataset(Dataset):    
    def __init__(self, data_root_folder, folder = '', n_sample=None):
        main_folder = os.path.join(data_root_folder, folder)
        self.folder_path = os.path.join(main_folder, 'slice')
        self.file_names = [f for f in os.listdir(self.folder_path)] #if f.endswith('.npy')]
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

        y_indices, x_indices = np.where(gt2D >= 0)  #
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
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    

if __name__ == "__main__":
    print("Data Loading Intialized.")
    
    BATCH_SIZE = 2

    data_root_folder = '/mnt/disks/disk_dir/full_raw - Copy/'
    train_dataset = BraTSDataset(data_root_folder, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4) 
    val_dataset = BraTSDataset(data_root_folder, 'val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4) 
    print("Data Loading Successful.")

    #test_dataset = BraTSDataset(data_root_folder, 'test')
    #test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=e, num_workers=4)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="lora_only",
        target_modules=["qkv", "q_proj", "v_proj"],
    )

    device = "cuda:0"

    processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base") #.to(device)
    model = SamModel.from_pretrained("wanglab/medsam-vit-base")
    dice_metric = Dice().to(device)
    jaccard_metric = BinaryJaccardIndex().to(device)
    lr = 1e-3
    loss_fn = GeneralizedDiceFocalLoss(sigmoid=False).to(device)
    model = get_peft_model(model, peft_config)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    print("Model Loading Successful.")

    print_trainable_parameters(model)

    train_dice_score = []
    train_jaccard_score = []
    train_loss = []
    for epoch in range(10):
        model.train()
        for i, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            imgs = batch['image'] #.to(device)
            masks = batch['mask'].to(device)
            boxes = batch['box'] #.to(device)
            inputs = processor(imgs, input_boxes=boxes, do_normalize = False, do_rescale = False, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs, multimask_output=False)
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            # convert soft mask to hard mask
            medsam_seg_prob = F.interpolate(
                medsam_seg_prob,
                size=(240, 240),
                mode="bilinear",
                align_corners=False,
            )

            loss = loss_fn(medsam_seg_prob, masks)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss.append(loss.item())
            dice_score = dice_metric(medsam_seg_prob, masks)
            jaccard_score = jaccard_metric(medsam_seg_prob, masks)

            train_dice_score.append(dice_score.cpu().item())
            train_jaccard_score.append(jaccard_score.cpu().item())

            print(f"Loss after iteration {i}: {loss.item()}, Train Dice : {dice_score.item()}, Train Jaccard : {jaccard_score.item()}")
        
        print(f"Train: Loss after epoch {epoch}: {np.mean(train_loss)}, Dice : {np.mean(train_dice_score)}, Dice : {np.mean(train_jaccard_score)}")
        
        model.eval()
        val_dice_score = []
        val_jaccard_score = []
        val_loss = []       
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader)):
                imgs = batch['image'] #.to(device)
                masks = batch['mask'].to(device)
                boxes = batch['box'] #.to(device)
                inputs = processor(imgs, input_boxes=boxes, do_normalize = False, do_rescale = False, return_tensors="pt")
                outputs = model(**inputs, multimask_output=False)
                medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                # convert soft mask to hard mask
                medsam_seg_prob = F.interpolate(
                    medsam_seg_prob,
                    size=(240, 240),
                    mode="bilinear",
                    align_corners=False,
                )

                loss = loss_fn(medsam_seg_prob, masks)

                train_loss.append(loss.item())
                dice_score = dice_metric(medsam_seg_prob, batch['mask'])
                jaccard_score = jaccard_metric(medsam_seg_prob, batch['mask'])

                val_dice_score.append(dice_score.cpu().item())
                val_jaccard_score.append(jaccard_score.cpu().item())

                print(f"Loss after iteration {i}: {loss.item()}, Val Dice : {dice_score.item()}, Val Jaccard : {jaccard_score.item()}")

        print(f"Val: Loss after epoch {epoch}: {np.mean(train_loss)}, Val Dice : {np.mean(val_dice_score)}, Val Jaccard : {np.mean(val_jaccard_score)}")


        torch.save(model.state_dict(), f'/mnt/disks/disk_dir/checkpoints/medsam_epoch_{epoch+1:03}.pth')
