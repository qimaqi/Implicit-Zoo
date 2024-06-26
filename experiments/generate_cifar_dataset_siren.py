import numpy as np
import os,sys,time
import torch
import tqdm
import PIL.Image,PIL.ImageDraw
import imageio
import hydra
import random

import torchvision
import torchvision.transforms as transforms
from src.model_zoo.siren import INR
from src.common import utils


def L1_loss(pred,label=0):
    loss = (pred.contiguous()-label).abs()
    return loss.mean()

def summarize_loss(opt,loss):
    loss_all = 0.
    assert("all" not in loss)
    # weigh losses
    for key in loss:
        assert(key in opt.loss_weight)
        assert(loss[key].shape==())
        if opt.loss_weight[key] is not None:
            assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
            assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
            loss_all += 10**float(opt.loss_weight[key])*loss[key]
    loss.update(all=loss_all)
    return loss

def MSE_loss(pred,label=0):
    loss = (pred.contiguous()-label)**2
    return loss.mean()

def generation(cfg):
    # dataloader 
    inv_normalize = transforms.Compose([ transforms.Normalize(mean = torch.zeros_like(torch.tensor(cfg.data.mean)),
                                                        std = 1/torch.tensor(cfg.data.std)),
                                    transforms.Normalize(mean = -torch.tensor(cfg.data.mean),
                                                        std = torch.ones_like(torch.tensor(cfg.data.std))),
                                ])
    

    transform = transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = torch.tensor(cfg.data.mean),
                                                        std = torch.tensor(cfg.data.std))
                                ])
    
    # change the download location if need
    trainset = torchvision.datasets.CIFAR10(root='./data', train=cfg.data.train,
                                            download=True,  transform= transform)
    
    if cfg.data.end == -1 or cfg.data.end > len(trainset):
        cfg.data.end = len(trainset)

    subset_indices = list(range(cfg.data.start, cfg.data.end))
    trainset = torch.utils.data.Subset(trainset, subset_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                            shuffle=False, num_workers=0)
    
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    save_dict = {}
    if cfg.seed.random_seed:
        seed_num = cfg.seed.seed_num
    else:
        seed_num = 1

    for img_index, data in tqdm.tqdm(enumerate(trainloader, 0),total=len(trainloader)): 
        inputs, labels = data
        inputs = inputs.to(cfg.device) 

        with torch.cuda.device(cfg.device):
            for seed_i in range(seed_num):
                if cfg.seed.random_seed:
                    random.seed(img_index+seed_i) # each image with different seed
                    np.random.seed(img_index+seed_i)
                    torch.manual_seed(img_index+seed_i)
                    torch.cuda.manual_seed_all(img_index+seed_i)
                    seed_name = str(img_index+seed_i).zfill(7)
                else:
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)
                    torch.cuda.manual_seed_all(42)
                    seed_name = str(42).zfill(7)

                psnr_i = 0
                eps = 0

                vis_path = "{}/vis/{}/".format(cfg.output_dir, str(labels.clone().detach().item()).zfill(2))
                os.makedirs(vis_path,exist_ok=True)
                ckpt_path = "{}/ckpts/{}/".format(cfg.output_dir, str(labels.clone().detach().item()).zfill(2))
                os.makedirs(ckpt_path,exist_ok=True)

                siren: INR = hydra.utils.instantiate(cfg.inrs,device=cfg.device)

                siren.to(cfg.device)
                optimizer = getattr(torch.optim,cfg.optim.algo)
                optim = optimizer([dict(params=siren.parameters(),lr=cfg.optim.lr)])
                
                # end when < 30 psnr, also forbid early stop, least 500 epochs
                while (psnr_i < 30 or eps<cfg.max_iter) and (eps<10*cfg.max_iter):
                    xy_grid = utils.make_coordinates(cfg.data.image_size, 1).to(cfg.device)
                    # utils.get_normalized_pixel_grid(cfg.data.image_size[0], cfg.data.image_size[1]).to(cfg.device)

                    optim.zero_grad()
                    rgb_est = siren.forward(xy_grid) # [B,HW,3]
                    rgb_est = rgb_est.view(cfg.data.image_size[0],cfg.data.image_size[1],3).permute(2,0,1)

                    loss = torch.nn.functional.mse_loss(inputs[0],rgb_est)
                    psnr_i = (-10*MSE_loss(inv_normalize(rgb_est),inv_normalize(inputs)).log10() )

                    loss.backward()
                    optim.step()
                    
                    eps+=1
              
                # save
                checkpoint = dict(
                image_idx=img_index,
                params=siren.state_dict())

                save_name = os.path.join(ckpt_path, f'{str(img_index).zfill(6)}_{seed_name}' + '.ckpt')
                torch.save(checkpoint,save_name)
                save_dict[str(img_index)] = psnr_i.item()

                if img_index % cfg.freq.vis == 0:
                    render_img = inv_normalize(siren.predict_entire_image()).clamp(min=0,max=1)        
                    frame = (render_img*255).byte().permute(1,2,0).numpy() 

                    imageio.imsave("{}/{}.png".format(vis_path,f'{str(img_index).zfill(6)}_{seed_name}'.zfill(7)),frame)
                    print("img_index psnr", img_index, 'on seed',seed_name ,psnr_i.item() , "on eps", eps)

            


    import json
    json_file_path = os.path.join(cfg.output_dir,f'{cfg.data.start}_{cfg.data.end}.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(save_dict, json_file)


@hydra.main(config_path="./cifar_generate_configs", config_name="main", version_base="1.1")
def main(cfg):
    from omegaconf import OmegaConf
    # pretty print hydra cfg
    print(OmegaConf.to_yaml(cfg, resolve=True))
    generation(cfg)


if __name__=="__main__":
    main()

