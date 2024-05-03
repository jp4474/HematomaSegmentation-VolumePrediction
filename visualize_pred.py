import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import SamModel, SamProcessor
import torch.nn.functional as F
import os
from torchmetrics.classification import BinaryJaccardIndex, Dice
from transformers import AutoConfig
from peft import get_peft_model, load_peft_weights, set_peft_model_state_dict, PeftConfig
from medsam_train_native import npyDataset
from segment_anything import sam_model_registry
from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from medsam_train_native import MedSAM_Lite
from ResAttUNet.scripts.models import ResAttU_Net
import cv2
import matplotlib.pyplot as plt
import copy


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

def get_bbox(mask, bbox_shift=3):
    y_indices, x_indices = np.where(mask > 0)
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes1024 = np.array([x_min, y_min, x_max, y_max])

    return bboxes1024

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

@torch.no_grad()
def make_prediction(model, type, img, mask, box):
    if type == "MedSAM":
        @torch.no_grad()
        def medsam_inference(medsam_model, img_embed, box_1024, H, W):
            box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

            sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = medsam_model.mask_decoder(
                image_embeddings=img_embed, # (B, 256, 64, 64)
                image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
                )

            low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

            low_res_pred = F.interpolate(
                low_res_pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, gt.shape)
            low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
            medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
            return medsam_seg
        # img must be 1024 x 1024 x 3
        # mask must be 1024 x 1024 x 1

        image_embedding = model.image_encoder(img)
        pred = medsam_inference(model, image_embedding, box, 512, 512)
    elif type == "LiteMedSAM" or type == "LoRA":
        outputs = model(img, box_np = box, mask = mask)
        logits = outputs['low_res_masks']
        pred = torch.sigmoid(logits)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
    elif type == "ResAttUNet":
        img = img[:,0,:,:].unsqueeze(1)
        assert img.shape == (1, 1, 512, 512), f"Image shape is {img.shape}"

        pred = model(img)
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
    
    return pred

class npyDataset(Dataset):    
    def __init__(self, data_root_folder = 'data', folder = 'train', n_sample=None):
        self.main_folder = os.path.join(data_root_folder, folder)
        self.imgs_path = os.path.join(data_root_folder, folder + '_final_npy/imgs')
        self.gts_path = os.path.join(data_root_folder, folder + '_final_npy/gts')
        temp1 =  [f for f in os.listdir(self.imgs_path) if f.endswith('.npy')]
        temp2 =  [f for f in os.listdir(self.gts_path) if f.endswith('.npy')]
        if n_sample is not None:
            self.imgs = sorted(temp1)[:n_sample]
            self.gts = sorted(temp2)[:n_sample]
        else:
            self.imgs = sorted(temp1)
            self.gts = sorted(temp2)

        assert len(self.imgs) == len(self.gts), "Number of images and masks should be same."

    def __getitem__(self, index):
        img_file_name = self.imgs[index]
        img = np.load(os.path.join(self.imgs_path, img_file_name), allow_pickle=True)

        img_256 = resize_longest_side(img, 256)
        img_256_tensor = torch.tensor(img_256).float().permute(2, 1, 0)

        img_512_tensor = torch.from_numpy(img).float().permute(2, 1, 0)

        img_1024 = resize_longest_side(img, 1024)
        img_1024_tensor = torch.tensor(img_1024).float().permute(2, 1, 0)
        #img_1024_tensor = img_1024_tensor.repeat(3, 1, 1)

        gt = np.load(os.path.join(self.gts_path, img_file_name), allow_pickle=True)
        gt[gt > 0] = 1

        gt_256 = resize_longest_side(gt, 256)
        gt_256_tensor = torch.from_numpy(gt_256).float().unsqueeze(0)

        gt_512_tensor = torch.from_numpy(gt).float().unsqueeze(0)
        gt_1024 = resize_longest_side(gt, 1024)
        gt_1024_tensor = torch.from_numpy(gt_1024).float().unsqueeze(0)

        box256 = get_bbox(gt_256)
        box512 = get_bbox(gt)
        box1024 = get_bbox(gt_1024)
        # box256 = get_bbox(box512, original_size=(256, 256))
        box256 = box256[None, ...] # (1, 4)
        box512 = box512[None, ...] # (1, 4)
        box1024 = box1024[None, ...] # (1, 4)

        return {
            'image_256': img_256_tensor,
            'mask_256': gt_256_tensor,
            'box_256': box256,
            'image_512': img_512_tensor,
            'mask_512': gt_512_tensor,
            'box_512': box512,
            'image_1024': img_1024_tensor,
            'mask_1024': gt_1024_tensor,
            'box_1024': box1024
        }
 
    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    # Load MedSAM
    MedSAM_CKPT_PATH = "medsam_vit_b.pth"
    MedSAM_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    MedSAM_model.eval()

    # Load LiteMedSAM
    LiteMedSAM_CKPT_PATH = "lite_medsam.pth"
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64) 
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    LiteMEDSAM_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    LiteMedSAM_CKPT = torch.load(LiteMedSAM_CKPT_PATH, map_location="cpu")
    LiteMEDSAM_model.load_state_dict(LiteMedSAM_CKPT)
    # Load LiteMedSAM + LoRA

    LoRA_model = copy.deepcopy(LiteMEDSAM_model)
    lora_config = PeftConfig.from_pretrained("output/LiteMedSAM/checkpoint-600")
    LoRA_model = get_peft_model(LoRA_model, lora_config)
    lora_weights = load_peft_weights("output/LiteMedSAM/checkpoint-600")
    set_peft_model_state_dict(LoRA_model, lora_weights)

    LiteMEDSAM_model.eval()
    LoRA_model.eval()

    # Load ResAttUNet
    ResAttUNet_model = ResAttU_Net(UnetLayer = 6, img_ch = 1)
    ResAttUNet_CKPT_PATH = "ResAttUNet/model_weights.pth"
    ResAttUNet_CKPT = torch.load(ResAttUNet_CKPT_PATH, map_location="cpu")
    ResAttUNet_model.load_state_dict(ResAttUNet_CKPT)
    ResAttUNet_model.eval()

    # DataLoader 
    dataset = npyDataset(data_root_folder='data', folder='test')
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, sample in enumerate(dataset_loader):
        img_256 = sample['image_256']
        mask_256 = sample['mask_256']
        box_256 = sample['box_256']
        img_512 = sample['image_512']
        mask_512 = sample['mask_512']
        box_512 = sample['box_512']
        img_1024 = sample['image_1024']
        mask_1024 = sample['mask_1024']
        box_1024 = sample['box_1024']
        
        resattunet_pred = make_prediction(ResAttUNet_model, "ResAttUNet", img_512, mask_512, box_512)
        litemedsam_pred = make_prediction(LiteMEDSAM_model, "LiteMedSAM",img_256, mask_256, box_256)
        medsam_pred = make_prediction(MedSAM_model,"MedSAM", img_1024, mask_1024, box_1024)
        lora_pred = make_prediction(LoRA_model, "LoRA", img_256, mask_256, box_256)

        # Metric Initialization
        def generate_title(name, dice, jaccard):
            return f'{name}\nDice: {round(dice, 2)}, Jaccard: {round(jaccard, 2)}'

        dice = Dice()
        jaccard = BinaryJaccardIndex()

        # print(mask_512.shape, resattunet_pred.shape)
        resattunet_dice = dice(resattunet_pred, mask_512[0].int()).item()
        medsam_dice = dice(torch.from_numpy(medsam_pred), mask_512.int()).item()
        litemedsam_dice = dice(litemedsam_pred, mask_256.int()).item()
        lora_dice = dice(lora_pred, mask_256.int()).item()

        resattunet_jaccard = jaccard(resattunet_pred, mask_512.int()).item()
        medsam_jaccard = jaccard(torch.from_numpy(medsam_pred), mask_512[0][0].int()).item()
        litemedsam_jaccard = jaccard(litemedsam_pred, mask_256.int()).item()
        lora_jaccard = jaccard(lora_pred, mask_256.int()).item()

        # Create a 2x3 grid for plotting
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        # Plot the original image
        axs[0, 0].imshow(img_512[0].numpy().transpose(1, 2, 0))
        axs[0, 0].set_title('Original Image')

        # Plot the original mask
        axs[0, 1].imshow(mask_256[0][0].numpy(), cmap='viridis')
        axs[0, 1].set_title('Original Mask')

        # Plot the predictions
        axs[0, 2].imshow(resattunet_pred[0][0], cmap='viridis')
        axs[0, 2].set_title(generate_title('ResAtt-UNet', resattunet_dice, resattunet_jaccard))

        axs[1, 0].imshow(litemedsam_pred[0].detach().numpy()[0], cmap='viridis')
        axs[1, 0].set_title(generate_title('TinyMedSAM', litemedsam_dice, litemedsam_jaccard))

        axs[1, 1].imshow(medsam_pred, cmap='viridis')
        axs[1, 1].set_title(generate_title('MedSAM', medsam_dice, medsam_jaccard))

        axs[1, 2].imshow(lora_pred[0].detach().numpy()[0], cmap='viridis')
        axs[1, 2].set_title(generate_title('TinyMedSAM+LoRA', lora_dice, lora_jaccard))

        # Remove the x and y ticks
        for ax in axs.flat:
            ax.axis('off')

        # Display the plot
        plt.show()