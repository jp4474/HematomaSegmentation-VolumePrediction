import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torchmetrics.classification import BinaryJaccardIndex, Dice
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
import numpy as np
import cv2
import argparse
from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from transformers.utils import logging

os.environ['PJRT_DEVICE'] = 'CUDA'

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


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
        #masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    

    
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

        gt = np.load(os.path.join(self.gts_path, img_file_name), allow_pickle=True)
        gt[gt > 0] = 1
        gt = resize_longest_side(gt, 256) # 512 -> 256
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0) # 512, 512
        box512 = get_bbox256(gt)
        box256 = resize_box_to_256(box512, original_size=(256, 256))
        box256 = box256[None, ...] # (1, 4)
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
        super().__init__(model, args)
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs['low_res_masks']
        masks = outputs['ground_truth_masks']        
        masks.requires_grad = True
        loss = loss_function(logits, masks)
        outputs = {'logits' : logits, 'masks' : masks}
        #metrics = compute_metrics(outputs)
        # metrics = {'train_' + key: value for key, value in metrics.items()}
        #self.log(metrics)
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        self.model.eval()
        val_loss = []
        dice_score = []
        jaccard_score = []

        for i, batch in enumerate(tqdm(eval_dataloader)):
            outputs = model(**batch)
            logits = outputs['low_res_masks']
            masks = outputs['ground_truth_masks']
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
        self.log({'val_loss' : val_loss, 'val_dice_score' : val_dice_score, 'val_jaccard_score' : val_jaccard_score})
        print({'val_loss' : val_loss, 'val_dice_score' : val_dice_score, 'val_jaccard_score' : val_jaccard_score})

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train MedSAM Lite model')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training')
parser.add_argument('--rank', type=int, default=32, help='Rank for LoraConfig')
parser.add_argument('--alpha', type=int, default=32, help='Alpha for LoraConfig')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for LoraConfig')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--use_rlora', type=bool, default=True, help='Use RLORA in LoraConfig')
parser.add_argument('--eval_steps', type=int, default=10, help='Number of steps to evaluate the model')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of gradient accumulation steps')
args = parser.parse_args()

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
lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cuda')
model.load_state_dict(lite_medsam_checkpoint)

# Use the arguments in the script
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
RANK = args.rank
ALPHA = args.alpha
DROPOUT = args.dropout
EPOCHS = args.epochs
USE_RLORA = args.use_rlora
EVAL_STEPS = args.eval_steps
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps

train_dataset = npyDataset(folder='train')
val_dataset = npyDataset(folder='val')

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    bias="lora_only",
    use_rslora=USE_RLORA,
    target_modules=["qkv, q_proj", "v_proj"], # train only q_proj and v_proj in mask decoder
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)
print(model.state_dict().keys())
model.train()
print("Model Loading Successful.")

model_name = 'LiteMedSAM'

training_args = TrainingArguments(
    output_dir=f"{model_name}-lora_{RANK}_{ALPHA}",
    learning_rate= LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=10,
    evaluation_strategy="steps",
    eval_steps = EVAL_STEPS,
    save_strategy="epoch",
    logging_steps=10,
    push_to_hub=False,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)   

trainer = SegmentationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()