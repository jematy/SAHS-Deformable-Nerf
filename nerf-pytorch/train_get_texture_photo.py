from nerf._init_spade import *
from nerf.texture_loader import *
import argparse
import glob
import os
import time
import sys
import numpy as np
import torch
import torchvision
import yaml
from nerf import cfgnode
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from nerf._init_spade import Generator as Generator
import itertools
import torch.optim as optim
from nerf import (CfgNode, mse2psnr)
import numpy as np
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)


def get_texture_identy_photo(mode, cfg):
    basedir = cfg.dataset.basedir
    fname = os.path.join(cfg.texture_refine.texture_photo)
    print("Done with data loading")
    img = cv2.cvtColor(cv2.imread(fname), code=cv2.COLOR_BGR2RGB)
    img = (np.array(img) / 255.0).astype(np.float32)
    H, W = img.shape[:2]
    img = cv2.resize(img, dsize=(H, W), interpolation=cv2.INTER_AREA)
    if cfg.nerf.train.white_background:
        img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
    return img, fname


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()
    cfg = None

    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    write = SummaryWriter(logdir)
    if cfg.dataset.type.lower() == "expression":
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from nerf import texture_loader
        training_data = texture_loader.Spade_NerfaceDataset(
            mode='train',
            cfg=cfg
        )
        training_data_val = texture_loader.Spade_NerfaceDataset(
            mode='val',
            cfg=cfg
        )
        output_data = texture_loader.Spade_NerfaceDataset_output(
            mode='train',
            cfg=cfg
        )
        val_data = texture_loader.Spade_NerfaceDataset_output(
            mode='val',
            cfg=cfg
        )
    elif cfg.dataset.type.lower() == "audio":
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from nerf import texture_loader
        training_data = texture_loader.Spade_AudioDataset(
            mode='train',
            cfg=cfg
        )
        training_data_val = texture_loader.Spade_AudioDataset(
            mode='val',
            cfg=cfg
        )
        output_data = texture_loader.Spade_AudioDataset_output(
            mode='train',
            cfg=cfg
        )
        val_data = texture_loader.Spade_AudioDataset_output(
            mode='val',
            cfg=cfg
        )
        # 将数组转换为图像
        # output_data[0][0] = np.array(output_data[0][0]) * 255
        # img = output_data[0][0]
        # # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #
        # # img = Image.fromarray(training_data[0][0], 'RGB')
        #
        # # # 展示图像
        # # img.show()
        # try:
        #     img_uint8 = img.astype(np.uint8)
        #     im_pil = Image.fromarray(img_uint8)
        #     im_pil.save('/data/jkx/level/output_image.jpg')
        #     print("Image saved successfully!")
        # except Exception as e:
        #     print("Error saving image:", e)
        # 如果需要，还可以将图像保存为文件
        # img.save('/data/jkx/output_image.png')
    print("done loading data")
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)
    G = Generator().to(device)
    G_opt = optim.Adam(G.parameters(), lr=cfg.texture_refine.lr_G,
                       betas=(cfg.texture_refine.beta1, cfg.texture_refine.beta2))
    start_epoch = 0
    # if True:
    #     checkpoint = "/data/jkx/level/logs/audio/Person_2/audio/person_2_se_fixed_bg_512_paper_model_texture_refine_right/checkpoint00110.ckpt"
    #     checkpoint = torch.load(checkpoint)
    #     G.load_state_dict(checkpoint["model_state_dict"])
    #     G_opt.load_state_dict(checkpoint["optimizer_state_dict"])
    #     start_epoch = 110
    if True:
        indenty_photo, frame_identy_photo = get_texture_identy_photo(
            mode='train',
            cfg=cfg
        )
    l1_norm = nn.L1Loss()
    l2_norm = nn.MSELoss()
    it = 0
    decayed_lr_G = cfg.texture_refine.lr_G
    total_epochs = cfg.texture_refine.epochs + cfg.texture_refine.epochs_decay

    # for epoch in tqdm(range(total_epochs), desc="Training", unit="epoch"):
    for epoch in tqdm(range(start_epoch, total_epochs), desc="Training", unit="epoch"):
        print("epoch: {}".format(epoch))
        if epoch >= cfg.texture_refine.epochs:
            decayed_lr_G = cfg.texture_refine.lr_G / cfg.texture_refine.epochs_decay * (total_epochs - epoch)
            set_lr(G_opt, decayed_lr_G)
        G.train()
        for i in range(len(training_data)):
            # print(output_data.__len__())
            image, _ = output_data[i]
            # image = image
            frame = indenty_photo
            frame = torch.from_numpy(frame).to(device)
            frame = frame.unsqueeze(0)
            frame = frame.permute(0, 3, 1, 2)
            image = torch.from_numpy(image).to(device)
            image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2)
            G_opt.zero_grad()
            fake_frame = G(frame, image)
            training_image = torch.from_numpy(training_data[i][0]).to(device)
            training_image = training_image.unsqueeze(0)
            training_image = training_image.permute(0, 3, 1, 2)
            if True:#to avoid negative values
                fake_frame = fake_frame.clamp(0, 1)
            # loss = l1_norm(fake_frame, frame) + l2_norm(fake_frame, frame) 写错了，要找到train的真实图片
            # loss = l2_norm(fake_frame, training_image) + l1_norm(fake_frame, training_image)
            loss = l2_norm(fake_frame, training_image)
            loss.backward()
            G_opt.step()
            write.add_scalar('loss', loss, it)
            write.add_scalar('lr', decayed_lr_G, it)
            if it % 100 == 0:
                print("epoch: {} iter: {} loss: {}".format(epoch, it, loss.item()))
            if not os.path.exists(os.path.join(logdir)):
                os.makedirs(os.path.join(logdir))
            it += 1

        # idx = np.random.randint(0, len(training_data))
        # if True:
        if epoch % 2 == 0:
            psnr_rgb = 0
            for idx in range(cfg.texture_refine.val_num):
                image1, _ = val_data[idx]
                frame1 = indenty_photo
                frame1 = torch.from_numpy(frame1).to(device)
                frame1 = frame1.unsqueeze(0)
                frame1 = frame1.permute(0, 3, 1, 2)
                image1 = torch.from_numpy(image1).to(device)
                image1 = image1.unsqueeze(0)
                image1 = image1.permute(0, 3, 1, 2)
                training_image = torch.from_numpy(training_data_val[idx][0]).to(device)
                training_image = training_image.unsqueeze(0)
                training_image = training_image.permute(0, 3, 1, 2)
                with torch.no_grad():
                    generated_photo = G(frame1, image1)
                    if True:  # to avoid negative values
                        generated_photo = generated_photo.clamp(0, 1)
                psnr_rgb += l2_norm(generated_photo, training_image).item()
                generated_photo = generated_photo.permute(0, 2, 3, 1)
                generated_photo = generated_photo.cpu().numpy()
                generated_photo = generated_photo.squeeze(0)
                generated_photo = generated_photo * 255
                generated_photo = generated_photo.astype(np.uint8)
                generated_photo = np.array(generated_photo)
                generated_photo = generated_photo.transpose(2, 0, 1)
                write.add_image('fake_frame', generated_photo, it)
            psnr_rgb = mse2psnr(psnr_rgb / cfg.texture_refine.val_num)
            write.add_scalar("train/psnr_rgb", psnr_rgb, it)
        if epoch % 10 == 0 or epoch == total_epochs-1:
            checkpoint_dict = {
                "iter": it,
                "i_batch": cfg.texture_refine.batch_size,
                "model_state_dict": G.state_dict(),
                # "aud_state_dict": AudNet.state_dict(),
                # "audattnet_state_dict": AudAttNet.state_dict(),
                "optimizer_state_dict": G_opt.state_dict(),
                # "optimizer_Aud_state_dict": optimizer_Aud.state_dict() if optimizer_Aud is not None else None,
                # "optimizer_AudAtt_state_dict": optimizer_AudAtt.state_dict() if optimizer_AudAtt is not None else None,
                "loss": loss,
                # "psnr": psnr,
                # "lpips": lpips,
                "background": None
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(epoch).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")


if __name__ == '__main__':
    main()