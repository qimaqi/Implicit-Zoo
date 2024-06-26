# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

import os
import time
import hydra
import random 

import torchmetrics
from src.common.utils import progress_bar
from src.common.randomaug import RandAugment
from src.layers.static_token import Learnable_Patch
from sklearn.metrics import ConfusionMatrixDisplay

from src.dataloading.siren_dataset import SirenWeightDataset 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import sys
from einops import rearrange

from timm.models.layers import PatchEmbed
# from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# from pprint import pprint

# nodes, _ = get_graph_node_names(classify_model, tracer_kwargs={'leaf_modules': [PatchEmbed]})
# pprint(nodes)


def draw_confusion_matrix(predict, tests, cfg, epoch):
    predict_np = predict.detach().cpu().numpy()
    tests_np = tests.detach().cpu().numpy()
    np.save(f"{cfg.output_dir}/predict_{epoch}.npy", predict_np)
    np.save(f"{cfg.output_dir}/tests_{epoch}.npy", tests_np)

    # save for further eval 
    ConfusionMatrixDisplay.from_predictions(tests_np, predict_np)
    # plt.savefig('./heatmap/pixel.png')
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_image = np.array(fig.canvas.renderer.buffer_rgba())
    pil_image = Image.fromarray(rgba_image[:, :, :3])  # Extract RGB channels
    rgb_tensor = torch.tensor(np.array(pil_image)).permute(2, 0, 1)
    plt.close('all')

    return rgb_tensor


def draw_pixel_location_heatmap_patch(pixels, cfg, epoch):
    # pixels N,Patch_H, Patch_W, 2
    patch_num, patch_h, patch_w = pixels.shape[:3]
    # print("patch_num", patch_num)

    pixels_np = pixels.detach().cpu().numpy()
    np.save(f"{cfg.output_dir}/learned_pixels_{epoch}.npy", pixels_np)
    image_array = np.arange(patch_num) / patch_num # 0-1 color
    np.random.seed(666)
    permute_idx = np.random.permutation(patch_num)
    image_array = image_array[permute_idx]
    color_intensity = np.repeat(image_array, patch_h*patch_w, axis=0)

    # print("pixels", pixels[...,0].min(), pixels[...,0].max(), pixels[...,1].min(), pixels[...,1].max())
    plt.scatter(pixels_np[..., 0], pixels_np[..., 1], c=color_intensity, marker='.', label='Pixel Locations', cmap='magma')
    # plt.savefig('./heatmap/pixel.png')
    fig = plt.gcf()
    fig.canvas.draw()
    rgba_image = np.array(fig.canvas.renderer.buffer_rgba())
    pil_image = Image.fromarray(rgba_image[:, :, :3])  # Extract RGB channels
    rgb_tensor = torch.tensor(np.array(pil_image)).permute(2, 0, 1)
    plt.close('all')

    return rgb_tensor

def train(cfg):
    # parsers
    # take in args
    use_wandb = cfg.use_wandb
    if use_wandb:
        import wandb
        # https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw
    #     wandb.config = OmegaConf.to_container(
    #     cfg, resolve=True, throw_on_missing=True
    # )

        watermark = "{}_patch_type_{}_seed{}_lr{}".format(cfg.output_dir, cfg.static_token.patch_type, cfg.seed ,cfg.token_lr)
        wandb.init(project="cifar10-challange_final",
                name=watermark, 
                group = cfg.group,
                settings=wandb.Settings(start_method="thread"),
                config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
                )
                # Track hyperparameters and run metadata
        )
        # wandb.config.update( OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        print("##### watermark #######", watermark)

    else: 
        writer = SummaryWriter(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Write the dictionary to the YAML file
    with open("{}/{}".format(cfg.output_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
                ), yaml_file, default_flow_style=False)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    trainset: SirenWeightDataset = hydra.utils.instantiate(cfg.data, task="train", debug = cfg.debug, _recursive_=False)
    evalset: SirenWeightDataset = hydra.utils.instantiate(cfg.data, task="eval", debug = cfg.debug, _recursive_=False)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=1)

    evalloader = torch.utils.data.DataLoader(evalset, batch_size= cfg.batch_size,
                                            shuffle=False, num_workers=1)
    
    inv_normalize = transforms.Compose([ transforms.Normalize(mean = torch.zeros_like(torch.tensor(cfg.data.mean)),
                                                        std = 1/torch.tensor(cfg.data.std)),
                                    transforms.Normalize(mean = -torch.tensor(cfg.data.mean),
                                                        std = torch.ones_like(torch.tensor(cfg.data.std))),
                                ])
        
    def differentiable_normalize(x, mean, std):
        # Scale and shift the input tensor
        normalized_x = (x - mean) / (std )
        
        return normalized_x

    def differentiable_inv_normalize(x, mean, std):
        # Scale and shift the input tensor
        normalized_x = (x * std +  mean)
        
        return normalized_x

    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Model factory..
    print('==> Building Classifiy model..')
    H,W = cfg.data.inrs.image_size
    classify_model = hydra.utils.instantiate(cfg.net)

    # token factory
    learnable_patch =hydra.utils.instantiate(cfg.static_token)

    if cfg.data.augmentation:
        color_aug = RandAugment(cfg.data.augmentation.N, cfg.data.augmentation.M, aug_tupe=2)
    



    # parameters_to_optimize =  [learable_token]
    parameters_to_optimize = [{"params": learnable_patch.learable_token, "lr": cfg.token_lr},{"params": list(classify_model.parameters()), "lr": cfg.clf_lr}] 
    if cfg.static_token.patch_type == 1:
        print("learnable_patch.scale_param", learnable_patch.scale_param)
    if cfg.static_token.patch_scale:
        parameters_to_optimize.append({"params": learnable_patch.scale_param, "lr": cfg.token_lr})
        # print("parameters_to_optimize", parameters_to_optimize)
  
    optimizer = hydra.utils.instantiate(cfg.opt, parameters_to_optimize)
    scheduler = hydra.utils.instantiate(cfg.sched, optimizer=optimizer)


    if cfg.resume:
        # TODO resume both clf model and learnable token
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(cfg.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'{cfg.resume}/checkpoint.pt')
        learnable_patch.load_state_dict(checkpoint['patch'])
        classify_model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    ##### Training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)   
    loss_func_cls = torch.nn.CrossEntropyLoss() 

    # For Multi-GPU
    if 'cuda' in cfg.device:
        print(cfg.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            classify_model = torch.nn.DataParallel(classify_model)
            cudnn.benchmark = True


    def train_epoch(epoch):
        print('\nEpoch: %d' % epoch)
        classify_model.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(trainloader, 0): 
            batch = batch.to(cfg.device)
            inputs_weight, input_bias, labels, masks  = batch.weights, batch.biases, batch.label, batch.mask
            # mask track the augmentation happened area

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                rgb_frames, draw_pixels = learnable_patch.patch2imgs(inputs_weight, input_bias, masks, to_patch = False) # rgb_patches

                # in training, if exist augmentation we want to also apply color augmentation
                # B x patch_num x h x w x 3
                if cfg.data.augmentation:
                    # rgb_debug = rearrange(rgb_patches, 'b (h w) (p1 p2 c) -> b  c (h p1) (w p2)', b = labels.shape[0], h = 32//cfg.static_token.patch_size, w = 32//cfg.static_token.patch_size, p1 = cfg.static_token.patch_size, p2 = cfg.static_token.patch_size, c = 3)
                    rgb_patches_inv = differentiable_inv_normalize(rgb_frames, torch.tensor(cfg.data.mean).to(rgb_frames.device).reshape(1,3,1,1), torch.tensor(cfg.data.std).to(rgb_frames.device).reshape(1,3,1,1) )
        
                    rgb_patches_aug = color_aug(rgb_patches_inv)

                    rgb_patches_aug = differentiable_normalize(rgb_patches_aug, torch.tensor(cfg.data.mean).to(rgb_frames.device).reshape(1,3,1,1), torch.tensor(cfg.data.std).to(rgb_frames.device).reshape(1,3,1,1))
                    rgb_frames = rgb_patches_aug


                est_class = classify_model(rgb_frames)
                cls_loss = loss_func_cls(est_class, labels)

                if cfg.static_token.patch_reg:
                    # we want the token to be away from each other
                    reg_loss = learnable_patch.cal_reg_loss()
                    cls_loss += reg_loss


            scaler.scale(cls_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += cls_loss.item()
            predicted = torch.argmax(est_class, dim = -1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # debug
            # print("learnable_patch.scale_param", learnable_patch.scale_param)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        return train_loss/(batch_idx+1)

    ##### Validation
    def eval_epoch(epoch, best_acc):
        classify_model.eval()
        eval_loss = 0
        correct = 0
        total = 0
        labels_plot = []
        preds_plot = []

        with torch.no_grad():
            
            for batch_idx, batch in enumerate(evalloader, 0): 
                batch = batch.to(cfg.device)
                inputs_weight, input_bias, labels, masks  = batch.weights, batch.biases, batch.label, batch.mask

                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    rgb_frames, draw_pixels = learnable_patch.patch2imgs(inputs_weight, input_bias)

                    est_class = classify_model(rgb_frames)
                    cls_loss = loss_func_cls(est_class, labels)
                    # # stack the imgs

                eval_loss += cls_loss.item()
                predicted = torch.argmax(est_class, dim = -1)

                accuracy.update(predicted.cpu(), labels.cpu())
                precision.update(predicted.cpu(), labels.cpu())
                recall.update(predicted.cpu(), labels.cpu())
                f1.update(predicted.cpu(), labels.cpu())
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                labels_plot.append(labels.flatten())
                preds_plot.append(predicted.flatten())

                progress_bar(batch_idx, len(evalloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (eval_loss/(batch_idx+1), 100.*correct/total, correct, total))

            labels_plot = torch.hstack(labels_plot)
            preds_plot = torch.hstack(preds_plot)
            heatmap_plot = draw_pixel_location_heatmap_patch(draw_pixels, cfg, epoch)
            confusion_plot = draw_confusion_matrix(preds_plot, labels_plot, cfg, epoch)    
            rgb_gt = learnable_patch.frame2imgs(inputs_weight, input_bias)[0:4]
            rgb_gt = inv_normalize(rgb_gt).clamp(0,1)
            img_vis = rgb_frames[0:4]
            img_vis = inv_normalize(img_vis).clamp(0,1)
            comprison = torch.concat([img_vis, rgb_gt], dim=0).detach().cpu()

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                "patch": learnable_patch.state_dict(),
                "model": classify_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), # "scaler": scaler.state_dict(),
                "scaler": scaler.state_dict(),
                "acc": best_acc,
                'epoch': epoch}
            
            save_path = os.path.join(cfg.output_dir, f'checkpoint.pt')
            torch.save(state, save_path)
            best_acc = acc
        
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {eval_loss:.5f}, acc: {(acc):.5f}'
        print(content)

        return eval_loss, acc, heatmap_plot, confusion_plot, best_acc, comprison

    list_loss = []
    list_acc = []

        
    classify_model.cuda()
    accuracy = torchmetrics.Accuracy(task='multiclass',average='micro', num_classes=10)
    precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=10)
    recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=10)
    f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=10)

    for epoch in range(start_epoch, cfg.epoch):
        start = time.time()
        trainloss= train_epoch(epoch)

        scheduler.step() # step cosine scheduling
        if epoch % 10 == 0 or epoch == cfg.epoch - 1:

            accuracy = torchmetrics.Accuracy(task='multiclass',average='micro', num_classes=10)
            precision = torchmetrics.Precision(task='multiclass', average='macro', num_classes=10)
            recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=10)
            f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=10)

            val_loss, acc, heatmap_plot, confusion_plot, best_acc, comprison = eval_epoch(epoch, best_acc)
            
            list_loss.append(val_loss)
            list_acc.append(acc)

            # est_recon_to_show = rgb_est[:4]
            # est_recon_to_show = inv_normalize(est_recon_to_show).clamp(0,1)
            # est_recon_to_show = torchvision.utils.make_grid(est_recon_to_show)
            
            # gt_to_show = rgb_gt[:4]
            # gt_to_show = inv_normalize(gt_to_show).clamp(0,1)
            # gt_to_show = torchvision.utils.make_grid(gt_to_show)  
            grid_img = torchvision.utils.make_grid(comprison, nrow=4).clamp(0,1)
            
            # Log training..
            if use_wandb:
                wandb.log({'epoch': epoch, 
                           'train_loss': trainloss, 
                           'val_loss': val_loss, 
                           "val_acc": acc, 
                           "token_lr": optimizer.param_groups[0]["lr"],
                            "Accuracy": accuracy.compute().item(),
                            "Precision": precision.compute().item(),
                            "Recall": recall.compute().item(),
                            "F1 Score": f1.compute().item(),
                            "clf_lr": optimizer.param_groups[-1]["lr"],
                            "epoch_time": time.time()-start}, step=epoch)
                # gt_to_show_np = gt_to_show.permute(1, 2, 0).cpu().numpy()
                # est_to_show_np = est_recon_to_show.permute(1, 2, 0).cpu().numpy()
                heatmap_plot_np = heatmap_plot.permute(1, 2, 0).cpu().numpy()
                confusion_plot_np = confusion_plot.permute(1, 2, 0).cpu().numpy()
                att_vis_plot_np = grid_img.permute(1, 2, 0).numpy()
                # wandb.log({"eval_gt_images": [wandb.Image(gt_to_show_np, caption="Eval GT Images")]}, step=epoch)
                # wandb.log({"eval_est_images": [wandb.Image(est_to_show_np, caption="Eval est Images")]}, step=epoch)
                wandb.log({"2D Heatmap": [wandb.Image(heatmap_plot_np, caption="Heatsmap")]}, step=epoch)
                wandb.log({"Confusion Matrix": [wandb.Image(confusion_plot_np, caption="confusion_matrix")]}, step=epoch)
                wandb.log({"Images compare": [wandb.Image(att_vis_plot_np, caption="Images Network see")]}, step=epoch)


            else:
                # writer.add_image('Eval gt image', gt_to_show, epoch)  
                # writer.add_image('Eval recon image', est_recon_to_show, epoch)    
                writer.add_image('confusion matrix', confusion_plot, epoch)  
                writer.add_image('2D Heatmap', heatmap_plot,epoch, dataformats='CHW')
                writer.add_scalar('Train loss', trainloss, epoch)
                writer.add_scalar('Eval loss', val_loss, epoch)
                writer.add_scalar('Eval accuracy %',  acc , epoch)
                writer.add_scalar('Accuracy %', accuracy.compute().item(), epoch)
                writer.add_scalar('Precision %',  precision.compute().item(), epoch)
                writer.add_scalar('Recall %', recall.compute().item(), epoch)
                writer.add_scalar('F1 Score %', f1.compute().item(), epoch)
                writer.add_scalar('token_lr', optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar('clf_lr', optimizer.param_groups[-1]["lr"], epoch)
                # Create a grid of images
                # print("grid_img", grid_img.size())
                writer.add_image('Images Compare', grid_img, dataformats='CHW')

            accuracy = torchmetrics.Accuracy(task='multiclass',average='micro', num_classes=10)
            precision = torchmetrics.Precision(task='multiclass', average='micro', num_classes=10)
            recall = torchmetrics.Recall(task='multiclass', average='micro', num_classes=10)
            f1 = torchmetrics.F1Score(task='multiclass', average='micro', num_classes=10)

            # writer.add_image('Attention_score', att_vis, epoch, dataformats='CHW')

    # writeout wandb
    if use_wandb:
        wandb.finish()
        # wandb.save("wandb_{}.h5".format(args.net))
    

@hydra.main(config_path="./train_cifar10_weight", config_name="main", version_base="1.1")
def main(cfg):
    # pretty print hydra cfg
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__=="__main__":
    main()

