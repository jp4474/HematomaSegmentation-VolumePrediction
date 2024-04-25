import os
import glob
from monai.losses import GeneralizedDiceFocalLoss
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import BinaryJaccardIndex, Dice
from peft import LoraConfig, get_peft_model, LoraModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from segment_anything import sam_model_registry
from transformers import SamModel, SamProcessor, TrainingArguments, Trainer
from datasets import load_metric
import evaluate
from pathlib import Path
import nibabel as nib
import numpy as np
import cv2
from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

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

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np, mask):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return {'low_res_masks' : low_res_masks, 'ground_truth_masks' : mask}

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
class SliceDataset:
    def __init__(self, img_pathes: Path, mask_pathes: Path, intensity_min, intensity_max) -> None:
        self.img_pathes = img_pathes
        self.mask_pathes = mask_pathes
        self.slices = [nib.load(p).shape[-1] for p in self.img_pathes]
        self.cum_slices = np.cumsum(self.slices)
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        #self.processor = processor
        

    def __getitem__(self, index: int):
        path_index = np.searchsorted(self.cum_slices, index, side='right')
        if path_index == 0:
            slice_index = index
        else:
            slice_index = index - self.cum_slices[path_index - 1]

        img = nib.load(self.img_pathes[path_index]).get_fdata()[:,:,slice_index]
        img = windowing(img, self.intensity_min, self.intensity_max)[np.newaxis, ...] # minmax scaler
        # mask = np.load(self.mask_pathes[path_index])[:,:,slice_index][np.newaxis, ...]
        mask = nib.load(self.mask_pathes[path_index]).get_fdata()[:,:,slice_index]
        bbox = compute_bounding_box(mask)

        # TODO: 
        # inputs = self.processor(img, input_boxes=bbox, do_normalize = False, do_rescale = False, return_tensors="pt")

        inputs['mask'] = mask.to(dtype=torch.float32)
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
        inputs['input_boxes'] = inputs['input_boxes'].squeeze(0)
        inputs = inputs.to(dtype=torch.float32)
        return inputs
        # return img.astype(np.float32), mask.astype(np.float32)
    
    def __len__(self):
        return self.cum_slices[-1]

class VolumeDataset:
    def __init__(self, img_pathes: Path, mask_pathes: Path, intensity_min, intensity_max) -> None:
        self.img_pathes = img_pathes
        self.mask_pathes = mask_pathes
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def __getitem__(self, index: int):
        path_index = index

        img = nib.load(self.img_pathes[path_index]).get_fdata()
        img = np.transpose(img, (2,0,1)) # why transposing?
        img = windowing(img, self.intensity_min, self.intensity_max)[:, np.newaxis, ...]
        mask = np.load(self.mask_pathes[path_index])
        mask = np.transpose(mask, (2,0,1))[:, np.newaxis, ...]

        return img.astype(np.float32), mask.astype(np.float32), path_index
    
    def __len__(self):
        return len(self.img_pathes)


def windowing(image, min_value, max_value):
    image_new = np.clip(image, min_value, max_value)
    image_new = (image_new - min_value) / (max_value - min_value)
    return image_new

class BraTSDataset(Dataset):    
    def __init__(self, data_root_folder, folder = '', n_sample=None):
        main_folder = os.path.join(data_root_folder, folder)
        self.folder_path = os.path.join(main_folder, 'slice')
        self.file_names = [f for f in os.listdir(self.folder_path) if f.endswith('.npy')]
        #self.file_names = sorted(os.listdir(self.folder_path))[:n_sample]

    def __getitem__(self, index):
        file_name = self.file_names[index]
        sample = np.load(os.path.join(self.folder_path, file_name), allow_pickle=True)
        img = sample[0,:,:]
        gt = sample[1,:,:]
        img_256 = resize_longest_side(img, 256)
        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256_padded = pad_image(img_256_norm, 256)
        img_256_tensor = torch.tensor(img_256_padded).float().unsqueeze(-1).permute(2, 0, 1) #.unsqueeze(0)
        img_256_tensor = torch.repeat_interleave(img_256_tensor, 3, dim=0)
        gt[gt > 0] = 1
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
        box = get_bbox256(gt)
        box256 = resize_box_to_256(box, original_size=(256, 256))
        box256 = box256[None, ...] # (1, 4)

        return {
            'image': img_256_tensor,
            'mask': gt_tensor,
            'box_np': box256
        }
        
    def __len__(self):
        return len(self.file_names)
    
class npyDataset(Dataset):    
    def __init__(self, data_root_folder = 'data', folder = 'train', n_sample=None):
        self.main_folder = os.path.join(data_root_folder, folder)
        self.imgs_path = os.path.join(data_root_folder, 'npy_' + folder + '/imgs')
        self.gts_path = os.path.join(data_root_folder, 'npy_' + folder + '/gts')
        if n_sample is not None:
            self.imgs = sorted(os.listdir(self.imgs_path))[:n_sample]
            self.gts = sorted(os.listdir(self.gts_path))[:n_sample]
        else:
            self.imgs = sorted(os.listdir(self.imgs_path))
            self.gts = sorted(os.listdir(self.gts_path)) 

        assert len(self.imgs) == len(self.gts), "Number of images and masks should be same."

    def __getitem__(self, index):
        img_file_name = self.imgs[index]
        img = np.load(os.path.join(self.imgs_path, img_file_name), allow_pickle=True)
        img_256 = resize_longest_side(img, 256)
        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256_padded = pad_image(img_256_norm, 256)
        img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1) #.unsqueeze(0)

        gt_file_name = self.gts[index]
        gt = np.load(os.path.join(self.gts_path, gt_file_name), allow_pickle=True)

        gt[gt > 0] = 1
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)
        box = get_bbox256(gt)
        box256 = resize_box_to_256(box, original_size=(256, 256))

        box256 = box256[None, ...] # (1, 4)

        # print(img_256_tensor.shape)
        # print(gt_256_tensor.shape)
        # print(box256.shape)

        return {
            'image': img_256_tensor,
            'mask': gt_tensor,
            'box_np': box256
        }
 
    def __len__(self):
        return len(self.imgs)


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

def compute_metrics(outputs):
    with torch.no_grad():
        dice_metric = Dice().to(outputs['logits'].device)
        jaccard_metric = BinaryJaccardIndex().to(outputs['logits'].device)

        logits, masks = outputs['logits'], outputs['masks'].int()

        return {'dice_metric' : dice_metric(logits, masks).item(), 
                'jaccard_metric' : jaccard_metric(logits, masks).item()}


def loss_function(logits, true_masks):
    return nn.BCEWithLogitsLoss()(logits, true_masks)   

class SegmentationTrainer(Trainer):
    def __init__(self, model, compute_metrics, args, train_dataset, eval_dataset):
        #super().__init__(model, args)
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['low_res_masks']
        masks = outputs['ground_truth_masks']
        logits = model.postprocess_masks(logits, (256, 256), (masks.shape[2], masks.shape[3])) #.squeeze(1)
        
        masks.requires_grad = True

        loss = loss_function(logits, masks) #self.loss_fn(logits, masks) 
        #logits.requires_grad = True
        outputs = {'logits' : logits, 'masks' : masks}
        metrics = compute_metrics(outputs)
        metrics = {'train_' + key: value for key, value in metrics.items()}
        self.log(metrics)
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        val_loss = []
        dice_score = []
        jaccard_score = []

        for i, batch in enumerate(tqdm(eval_dataloader)):
            outputs = model(**batch)
            logits = outputs['low_res_masks']
            masks = outputs['ground_truth_masks']
            logits = model.postprocess_masks(logits, (256, 256), (masks.shape[2], masks.shape[3])) #.squeeze(1)
            masks = outputs['ground_truth_masks']
            loss = loss_function(logits, masks)
            outputs = {'logits' : logits, 'masks' : masks}
            metrics = compute_metrics(outputs)
            val_loss.append(loss.item())
            dice_score.append(metrics['dice_metric'])
            jaccard_score.append(metrics['jaccard_metric'])
        
        val_loss = np.mean(val_loss)
        val_dice_score = np.mean(dice_score)
        val_jaccard_score = np.mean(jaccard_score)
        print({'val_loss' : val_loss, 'val_dice_score' : val_dice_score, 'val_jaccard_score' : val_jaccard_score})
        #return {'val_loss' : val_loss, 'val_dice_score' : val_dice_score, 'val_jaccard_score' : val_jaccard_score}

if __name__ == "__main__":
    print("Data Loading Intialized.")
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

    model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )

    lite_medsam_checkpoint_path = os.path.join(os.getcwd(), 'lite_medsam.pth')
    lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
    model.load_state_dict(lite_medsam_checkpoint)

    BATCH_SIZE = 1
    LEARNING_RATE = 0.00005
    RANK = 32
    ALPHA = 32
    DROPOUT = 0.1
    EPOCHS = 10
    USE_RLORA = True

    train_dataset = npyDataset(folder='train')
    val_dataset =  npyDataset(folder='val')
    
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="lora_only",
        use_rslora=USE_RLORA,
        target_modules=["q_proj", "v_proj"], # train only q_proj and v_proj in mask decoder
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    print("Model Loading Successful.")

    model_name = 'LiteMedSAM'

    training_args = TrainingArguments(
        output_dir=f"{model_name}-lora_{RANK}_{ALPHA}",
        learning_rate= LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps = 1,
        save_strategy="epoch",
        logging_steps=10,
        push_to_hub=False
    )   

    trainer = SegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    #trainer.train()
    trainer.evaluate()
