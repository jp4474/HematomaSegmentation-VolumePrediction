import configparser
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import torch
from torch import optim, Tensor
from torch.nn import (
    functional as F,
    Module,
    utils
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import models
from scripts.dataset import SliceDataset, VolumeDataset
from scripts.utils import get_memory_usage, TqdmExtraFormat
from time import time
from torchmetrics.classification import BinaryJaccardIndex, Dice, JaccardIndex

def main():
    start_time = time()
    config = configparser.ConfigParser()
    config.read('config.ini')
    visualize(config)
    # test(config)
    end_time = time()
    print("Runtime: ", end_time - start_time)

def visualize(config: configparser.ConfigParser):
    print('visualizing...')
    img_train_val_test = Path(config['train']['img_val'])
    mask_train_val_pathes = sorted(Path(config['train']['mask_val']).glob('*.npy'))
    img_train_val_pathes = [img_train_val_test / p.name.replace('.npy', '.nii.gz') for p in mask_train_val_pathes]
    print(f'len(img_train_val_pathes) = {len(img_train_val_pathes)}')
    print(f'len(mask_train_val_pathes) = {len(mask_train_val_pathes)}')
    # train_size = config['train'].getfloat('train_size')
    # random_seed = config['train'].getint('random_seed')
    # img_train_pathes, img_val_pathes, mask_train_pathes, mask_val_pathes = train_test_split(
    #     img_train_val_pathes, mask_train_val_pathes, train_size=train_size, random_state=random_seed
    # )
    img_train_pathes = img_train_val_pathes
    mask_train_pathes = mask_train_val_pathes
    # print(f'len(img_train_pathes) = {len(img_train_pathes)}')
    # print(f'len(img_val_pathes) = {len(img_val_pathes)}')
    # print(f'len(mask_train_pathes) = {len(mask_train_pathes)}')
    # print(f'len(mask_val_pathes) = {len(mask_val_pathes)}')
    intensity_min = config['train'].getint('intensity_min')
    intensity_max = config['train'].getint('intensity_max')
    trainset = SliceDataset(img_train_pathes, mask_train_pathes, intensity_min, intensity_max)
    # valset = VolumeDataset(img_val_pathes, mask_val_pathes, intensity_min, intensity_max)
    print(f'len(trainset) = {len(trainset)}')
    # print(f'len(valset) = {len(valset)}')
    trainloader = DataLoader(trainset, **eval(config['train']['trainloader_kwargs']))
    # valloader = DataLoader(valset, batch_size=None, shuffle=False, num_workers=1)
    print(f'len(trainloader) = {len(trainloader)}')
    # print(f'len(valloader) = {len(valloader)}')
    model: Module = getattr(models, config['test']['model_name'])(**eval(config['train']['model_kwargs']))
    print(f'model.__class__.__name__ = {model.__class__.__name__}')
    print()
    model.load_state_dict(torch.load(config['test']['save_path'], map_location = 'cuda:0'))
    # device = config['train']['device']
    device = "cuda:0"
    model.to(device)
    debug = config['train'].getboolean('debug')
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=1 / 10 ** .5, patience=2, verbose=True)
    val_metrics = Path(config['train']['val_metrics'])
    if val_metrics.is_file():
        val_metrics.unlink()
    for epoch in range(1):
        print(f'epoch = {epoch:03d}')
        visualize_epoch(config, trainloader, model, optimizer)
        print('GPU_memory_usage :', get_memory_usage())
        # val_epoch(config, valloader, model, scheduler)
        # print('GPU_memory_usage :', get_memory_usage())
        # print()
        if debug and epoch == 2:
            break
        if optimizer.param_groups[0]['lr'] < 1e-6:
            break


def test(config: configparser.ConfigParser):
    test_dice_scores = []
    print('Testing...')
    img_train_val_test = Path(config['train']['img_train_val_test'])
    mask_test_pathes = sorted(Path(config['test']['mask_test']).glob('*.npy'))
    img_test_pathes = [img_train_val_test / p.name.replace('.npy', '.nii.gz') for p in mask_test_pathes]
    print(f'len(img_test_pathes) = {len(img_test_pathes)}')
    print(f'len(mask_test_pathes) = {len(mask_test_pathes)}')
    intensity_min = config['train'].getint('intensity_min')
    intensity_max = config['train'].getint('intensity_max')
    testset = VolumeDataset(img_test_pathes, mask_test_pathes, intensity_min, intensity_max)
    print(f'len(testset) = {len(testset)}')
    testloader = DataLoader(testset, batch_size= None, shuffle=False, num_workers=1)
    #TODO: change line 
    # testset = VolumeDataset(img_test_pathes, mask_test_pathes, intensity_min, intensity_max)
    # testloader = DataLoader(testset, batch_size= None, shuffle=False, num_workers=1)
    #TODO: change line 
    print(f'len(testloader) = {len(testloader)}')
    model: Module = getattr(models, config['test']['model_name'])(**eval(config['train']['model_kwargs']))
    print(f'model.__class__.__name__ = {model.__class__.__name__}')
    print()
    model.load_state_dict(torch.load(config['test']['save_path'], map_location = 'cuda:0'))
    # model = torch.load()
    device = "cuda:0"
    model.to(device)
    debug = config['train'].getboolean('debug')

    metrics = np.zeros(len(testloader))
    pdf = PdfPages('figures.pdf')
    test_metrics = Path(config['test']['test_metrics'])
    print("Start inference")
    with torch.no_grad(), tqdm(total=len(testloader)) as pbar:
        for ind, sample in enumerate(testloader):
            sample: tuple[Tensor, ...]
            img, mask_true, path_idx = sample
            model_output = torch.cat([model(slc.unsqueeze(0).to(device)) for slc in img])

            mask_pred = (model_output > 0).cpu().numpy()
            mask_true = (mask_true == 1).numpy()
            dice = 1 - distance.dice(mask_pred.reshape(-1), mask_true.reshape(-1))
            metrics[path_idx] = dice
            print(f"Dice for {ind}: ", dice)
            test_dice_scores.append(dice)
            pbar.update()
            fig = plot(img.numpy(), mask_true, mask_pred, path_idx)
            fig.set_size_inches(15, 5 * mask_true.shape[0])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            with open("test_dice_scores.txt", "w") as output:
                output.write(str(test_dice_scores))

            if debug and path_idx == 1:
                break
    pdf.close()
    np.savetxt(test_metrics, metrics, fmt='%.5f')


def visualize_epoch(config: configparser.ConfigParser, loader: DataLoader, model: Module, optimizer: optim.Adam):
    print(f'length of train loader = {len(loader)}')
    model.train()
    device = "cuda:0"
    debug = config['train'].getboolean('debug')

    loss_values = list()
    pbar = TqdmExtraFormat(
        total=14 * 60, dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {elapsed}<{remaining}/{total_time} [{rate_fmt}{postfix}]'
    )
    sigmoid = torch.nn.Sigmoid()
    for idx, sample in enumerate(loader):
        if idx < 3:
            continue
        sample: tuple[Tensor, ...]
        img, mask_true = sample

        # print("img shape", img.shape)
        # print("img type", type(img))
        # print("mask shape", mask_true.shape)
        # print("mask type", type(mask_true))

        model_output = model(img.to(device))
        # print("model_output shape", model_output.shape)
        # print("model_output type", type(model_output))

        mask = mask_true.cpu()[0][0]
        pred = torch.Tensor(model_output).cpu()[0][0]
        img = img.cpu()[0][0]
        pred = sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        # Flatten the mask and prediction arrays for score calculations
        dice_metric = Dice()
        jaccard = BinaryJaccardIndex()

        # print("mask type ", type(mask))
        # print("mask ", mask)
        # Calculate Dice and Jaccard scores
        # dice_score = dice_coeff_binary(pred, mask)
        # print("idx", idx)
        # print("pred shape", pred.shape)
        # print("mask shape", mask.shape)
        # print("img shape", img.shape)
        dice_score = dice_metric(pred, mask.long())
        jaccard_score = jaccard(pred, mask)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(img.permute(1, 2, 0))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        # ax[1].imshow(mask.permute(1, 2, 0))
        ax[1].imshow(mask)
        ax[1].set_title(f"Ground Truth")
        # ax[2].imshow(pred.permute(1, 2, 0))
        ax[2].imshow(pred.detach())
        ax[2].set_title(f"Prediction\nDice: {dice_score.cpu():.2f} Jaccard: {jaccard_score.cpu():.2f}")
        plt.show()
        plt.savefig(f"visualization/yangting_{idx}.png")


        # loss = F.binary_cross_entropy_with_logits(model_output, mask_true.to(device))

        # optimizer.zero_grad()
        # loss.backward()
        # loss_values.append(loss.item())
        # utils.clip_grad_norm_(model.parameters(), 0.25)
        # optimizer.step()
        # pbar.n = min(int(pbar.format_dict['elapsed']), pbar.total)
        # pbar.set_postfix(samples=(idx + 1) * loader.batch_size)
        # pbar.refresh()

        # if debug and idx == 1:
        #     break

        # if pbar.format_dict['elapsed'] > 14 * 60:  # training should be less than 15 mins
        #     break
    pbar.close()
    print(f'loss = {np.mean(loss_values)} ± {np.std(loss_values)}')


def val_epoch(config: configparser.ConfigParser, loader: DataLoader, model: Module, scheduler: optim.lr_scheduler.ReduceLROnPlateau):
    print(f'length of validation loader = {len(loader)}')
    model.eval()
    device = "cuda:0"
    debug = config['train'].getboolean('debug')
    save_path = Path(config['train']['save_path'])

    val_metrics = Path(config['train']['val_metrics'])
    if not val_metrics.is_file():
        old_metrics = np.zeros(len(loader))
    else:
        old_metrics = np.loadtxt(val_metrics)
    new_metrics = old_metrics.copy()

    pbar = TqdmExtraFormat(
        total=1 * 60, dynamic_ncols=True,
        bar_format='{l_bar}{bar}| {elapsed}<{remaining}/{total_time} [{rate_fmt}{postfix}]'
    )
    with torch.no_grad():
        for idx, sample in enumerate(loader):
            sample: tuple[Tensor, ...]
            img, mask_true, path_idx = sample
            model_output = torch.cat([model(slc.unsqueeze(0).to(device)) for slc in img])

            mask_pred = (model_output > 0).cpu().numpy()
            mask_true = (mask_true == 1).numpy()
            dice = 1 - distance.dice(mask_pred.reshape(-1), mask_true.reshape(-1))
            new_metrics[path_idx] = dice

            pbar.n = min(int(pbar.format_dict['elapsed']), pbar.total)
            pbar.set_postfix(samples=idx + 1)
            pbar.refresh()

            if debug and idx == 1:
                break

            if pbar.format_dict['elapsed'] > 1 * 60:  # training should be less than 15 mins
                break
    pbar.close()
    print(f'dice = {np.mean(new_metrics)} ± {np.std(new_metrics)}')
    if np.mean(new_metrics) > np.mean(old_metrics) or not save_path.is_file():
        torch.save(model.state_dict(), save_path)
    scheduler.step(np.mean(new_metrics))
    np.savetxt(val_metrics, new_metrics, fmt='%.5f')


def plot(img, mask_true, mask_pred, idx):
    img = np.squeeze(img, axis=1)
    mask_true = np.squeeze(mask_true, axis=1)
    mask_pred = np.squeeze(mask_pred, axis=1)
    max_value = img.max()
    img /= max_value

    num_slice = img.shape[0]
    fig, axs = plt.subplots(num_slice, 3)
    axs: list[list[plt.Axes]]
    for i in range(num_slice):
        axs[i][0].imshow(img[i], cmap='gray')
        axs[i][0].set_title(f'z = {i}')
        axs[i][1].imshow(mask_true[i], cmap='gray')
        axs[i][1].set_title(f'ID {idx:03d}\nGround truth')
        axs[i][2].imshow(mask_pred[i], cmap='gray')
        axs[i][2].set_title('AI generated')
    for i in range(num_slice):
        for j in range(3):
            axs[i][j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig


if __name__ == '__main__':
    main()
