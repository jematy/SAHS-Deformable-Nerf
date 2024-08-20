import argparse
import glob
import os
import time
import sys

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import torchvision
import yaml

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from nerf.load_flame import load_flame_data

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, get_ray_bundle_by_mask, img2mse,
                  load_llff_data, meshgrid_xy, models, utils, MaskMSELoss,
                  mse2psnr, run_one_iter_of_nerf, dump_rays, GaussianSmoothing, MaskCrossEntropyLoss)
# from nerf.metrics import lpips_single_image_pair_tensor, lpips_single_image_pair_tensor_2
# import lpips

# from gpu_profile import gpu_profile
# lpips_fn = lpips.LPIPS(net='alex').to(device='cuda')
def main():
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

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    # nosmo_iters, smo_size = 200000, 8
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf, expressions = None, None, None, None, None
        if cfg.dataset.type.lower() == "expression":
            from torch.utils.data import DataLoader
            from torch.utils.data import Dataset
            from nerf import nerface_dataloader
            training_data = nerface_dataloader.NerfaceDataset(
                mode='train',
                cfg=cfg
            )

            validation_data = nerface_dataloader.NerfaceDataset(
                mode='val',
                cfg=cfg,
            )

            H = training_data.H
            W = training_data.W


        if cfg.dataset.type.lower() == "audio":
            from torch.utils.data import DataLoader
            from torch.utils.data import Dataset
            from nerf import audio_dataloader
            training_data = audio_dataloader.AudioDataset(
                mode='train',
                cfg=cfg
            )

            validation_data = audio_dataloader.AudioDataset(
                mode='val',
                cfg=cfg,
                testskip=cfg.dataset.testskip,
            )

            H = training_data.H
            W = training_data.W
            # nosmo_iters = cfg.experiment.nosmo_iters
            # smo_size = cfg.experiment.smo_size

    print("done loading data")
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"  # + ":" + str(cfg.experiment.device)
    else:
        device = "cpu"

    model = getattr(models, cfg.models.mask.type)(cfg)
    model.to(device)
    # maskGenerator = getattr(models, cfg.models.mask.type)(cfg)
    # maskGenerator.to(device)

    ###################################
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = False  # True
    disable_driving = False  # True to disable expressions
    disable_latent_codes = True  # False # True to disable latent codes
    fixed_background = True # Do False to disable BG
    regularize_latent_codes = False  # True to add latent code LOSS, false for most experiments

    train_spatial_embeddings = True
    regularize_spatial_embedding = False

    dynamic_sampling = True
    ###################################
    ###################################

    supervised_train_background = train_background and supervised_train_background
    # Avg background
    # images[i_train]
    if train_background:  # TODO doesnt support dataloader!
        with torch.no_grad():
            avg_img = torch.mean(images[i_train], axis=0)
            # Blur Background:
            if blur_background:
                avg_img = avg_img.permute(2, 0, 1)
                avg_img = avg_img.unsqueeze(0)
                smoother = GaussianSmoothing(channels=3, kernel_size=11, sigma=11)
                print("smoothed background initialization. shape ", avg_img.shape)
                avg_img = smoother(avg_img).squeeze(0).permute(1, 2, 0)
            # avg_img = torch.zeros(H,W,3)
            # avg_img = torch.rand(H,W,3)
            # avg_img = 0.5*(torch.rand(H,W,3) + torch.mean(images[i_train],axis=0))
            background = torch.tensor(avg_img, device=device)
        background.requires_grad = True

    if fixed_background:  # load GT background
        print("loading GT background to condition on")
        from PIL import Image
        if cfg.dataset.type.lower() == "expression":
            background = Image.open(os.path.join(cfg.dataset.basedir, 'bg', '00050.png'))
        else:
            background = Image.open(os.path.join(cfg.dataset.basedir, 'bc.jpg'))
        background.thumbnail((H, W))
        background = torch.from_numpy(np.array(background).astype(np.float32))
        background = background[:, :, :3]
        background = background / 255

        background = torch.cat((background, torch.ones(H, W, 1), torch.zeros(H, W, 11)), dim=-1).to(device)
        print("bg shape", background.shape)
        # print("should be ", training_data[0].shape)
        # assert background.shape[:2] == [training_data.H,training_data.W]
    else:
        background = None

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # Initialize optimizer.
    trainable_parameters = list(model.parameters())
    if train_background:
        # background.requires_grad = True
        # trainable_parameters.append(background) # add it later when init optimizer for different lr
        print("background.is_leaf ", background.is_leaf, background.device)

    if train_latent_codes:
        latent_codes = torch.zeros(len(training_data), 32, device=device)
        # latent_codes_torso = torch.zeros(len(training_data), 32, device=device)
        print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
        if not disable_latent_codes:
            trainable_parameters.append(latent_codes)
            latent_codes.requires_grad = True
            # trainable_parameters.append(latent_codes_torso)
            # latent_codes_torso.requires_grad = True

    sample_prob = torch.ones(12, device=device)

    if train_background:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params': trainable_parameters},
             {'params': background, 'lr': cfg.optimizer.lr}],
            lr=cfg.optimizer.lr
        )
    else:
        optimizer = getattr(torch.optim, cfg.optimizer.type)(
            [{'params': trainable_parameters}],  # this is obsolete but need for continuing training
            lr=cfg.optimizer.lr
        )
    # optimizer_Aud, optimizer_AudAtt = None, None
    # AudNet, AudAttNet = None, None
    # if cfg.dataset.type.lower() == "audio":
    #     from nerf.modules import AudioNet, AudioAttNet
    #     AudNet = AudioNet(76, 16).to(device)
    #     AudAttNet = AudioAttNet().to(device)
    #     optimizer_Aud = getattr(torch.optim, cfg.optimizer.type)(
    #         [{'params': AudNet.parameters()}],  # this is obsolete but need for continuing training
    #         lr=cfg.optimizer.lr
    #     )
    #     optimizer_AudAtt = getattr(torch.optim, cfg.optimizer.type)(
    #         [{'params': AudAttNet.parameters()}],  # this is obsolete but need for continuing training
    #         lr=cfg.optimizer.lr
    #     )
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0
    i_batch = 0
    N_rand = 100


    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
        if checkpoint["latent_codes"] is not None:
            print("loaded latent codes from checkpoint")
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))
        if 'sample_prob' in checkpoint and checkpoint['sample_prob'] is not None:
            print("loaded sampling probability from checkpoint")


        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # if cfg.dataset.type.lower() == "audio":
        #     optimizer_Aud.load_state_dict(checkpoint["optimizer_Aud_state_dict"])
        #     optimizer_AudAtt.load_state_dict(checkpoint["optimizer_AudAtt_state_dict"])
        start_iter = checkpoint["iter"] + 1
        i_batch = checkpoint["i_batch"]

    print("Starting loop")
    # auds = None
    # if cfg.dataset.type.lower() == "audio":
    #     auds = training_data.get_all_auds()
    #     auds = torch.Tensor(auds).to(device).float()
    img_data, pose_data, driving_data, probs = None, None, None, None
    _, _, pose_c, _, _, _ = training_data[0]
    pose_c = torch.Tensor(pose_c[:3, :4]).to(device)
    sample_prob_weights = torch.ones(12, device=device)
    sample_prob_weights[7:9] = 2
    cross_entropy_loss = MaskCrossEntropyLoss(sample_prob_weights)
    mse_loss = MaskMSELoss(sample_prob_weights)

    for i in trange(start_iter, cfg.experiment.train_iters):

        model.train()
        # if cfg.dataset.type.lower() == "audio":
        #     AudNet.train()
        #     AudAttNet.train()

        background_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                expressions=expressions
            )
        else:
            # img_idx_all = i_batch
            # if i % N_rand == 0:
            #     if i_batch + N_rand <= len(training_data):
            #         img_data, mask_data, pose_data, hwk, driving_data, probs = \
            #             training_data[i_batch:i_batch + N_rand]  # 读取N_rand张图片
            #         i_batch += N_rand
            #     else:
            #         img_data, mask_data, pose_data, hwk, driving_data, probs = \
            #             np.concatenate((training_data[i_batch:],
            #                             training_data[:(i_batch + N_rand) % len(training_data)]), axis=1)
            #         i_batch = (i_batch + N_rand) % len(training_data)

            img_i = np.random.choice(len(training_data))
            img_data, mask_data, pose_data, hwk, driving_data, probs = training_data[img_i]
            # [H, W, focal] = hwk[0]
            # img_idx = np.random.choice(N_rand)
            # img_target = torch.Tensor(img_data[img_idx]).to(device)
            # mask_target = torch.BoolTensor(mask_data[img_idx] > 0.5).to(device)
            # pose_target = torch.Tensor(pose_data[img_idx][:3, :4]).to(device)
            # img_idx_all = (img_idx_all + img_idx) % len(training_data)
            [H, W, focal] = hwk
            img_target = torch.Tensor(img_data).to(device)
            mask_target = torch.from_numpy(mask_data).to(device)
            # mask_target = torch.IntTensor(mask_data > 0.2).to(device)
            pose_target = torch.Tensor(pose_data[:3, :4]).to(device)

            # 引入噪声
            # epsilon = 0.001
            # p = torch.rand(H, W) < epsilon
            # num = torch.count_nonzero(p)
            # indexes = torch.randint(mask_target.shape[-1], size=[num])
            # mask_target[torch.where(p)[0], torch.where(p)[1]] = torch.zeros(1,mask_target.shape[-1], dtype=torch.int32, device=mask_target.device)
            # mask_target[torch.where(p)[0], torch.where(p)[1], indexes] = 1

            driving_target_smo = None
            if not disable_driving:
                # if cfg.dataset.type.lower() == "audio":
                #     aud = auds[img_idx_all]
                #     if i >= nosmo_iters:
                #         smo_half_win = int(smo_size / 2)
                #         left_i = img_idx_all - smo_half_win
                #         right_i = img_idx_all + smo_half_win
                #         pad_left, pad_right = 0, 0
                #         if left_i < 0:
                #             pad_left = -left_i
                #             left_i = 0
                #         if right_i > len(training_data):
                #             pad_right = right_i - len(training_data)
                #             right_i = len(training_data)
                #         auds_win = auds[left_i:right_i]
                #         if pad_left > 0:
                #             auds_win = torch.cat(
                #                 (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                #         if pad_right > 0:
                #             auds_win = torch.cat(
                #                 (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                #         auds_win = AudNet(auds_win)
                #         driving_target = auds_win[smo_half_win]
                #         driving_target_smo = AudAttNet(auds_win)
                #     else:
                #         driving_target = AudNet(aud.unsqueeze(0))
                # else:
                #     driving_target = torch.Tensor(driving_data[img_idx]).to(device)  # vector

                # driving_target = torch.Tensor(driving_data[img_idx]).to(device)  # vector
                driving_target = torch.Tensor(driving_data).to(device)
            else:  # zero driving
                driving_target = torch.zeros(76, device=device)

            if not disable_latent_codes:
                # latent_code = latent_codes[img_idx_all].to(device) if train_latent_codes else None
                latent_code = latent_codes[img_i].to(device) if train_latent_codes else None
            else:
                latent_codes = torch.zeros(32, device=device)

            with torch.no_grad():
                probs = sample_prob.unsqueeze(0).unsqueeze(0) * mask_target
                probs = torch.sum(probs, dim=-1)
                probs = probs.cpu().numpy()
                probs /= probs.sum()
            x, y = meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device))

            # ray_origins, ray_directions = get_ray_bundle_by_mask(H, W, focal, pose_target, mask_head)

            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

            coords = torch.cat(
                (x.unsqueeze(-1), y.unsqueeze(-1), mask_target),
                dim=-1,
            )

            # Only randomly choose rays that are in the bounding box !
            # coords = torch.stack(
            #     meshgrid_xy(torch.arange(bbox[0],bbox[1]).to(device), torch.arange(bbox[2],bbox[3]).to(device)),
            #     dim=-1,
            # )

            coords = coords.reshape((-1, 1+1+mask_target.shape[-1]))
            # select_inds = np.random.choice(
            #     coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            # )

            # Use importance sampling to sample mainly in the bbox with prob p
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=probs.reshape(-1)
            )

            select_inds = coords[select_inds]
            inHead = select_inds[..., -mask_target.shape[-1]:]  # [2048, class_num]

            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]  # [2048, 3]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            # dump_rays(ray_origins, ray_directions)

            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (
                        train_background or fixed_background) else None
            mask_target = mask_target[select_inds[:, 0], select_inds[:, 1], :].float()


            print(driving_target.shape)
            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                driving=driving_target,  # driving_target_smo if cfg.dataset.type.lower() == "audio" and i >= nosmo_iters else driving_target,
                pose=pose_target,
                pose_c=pose_c,
                background_prior=background_ray_values,
                latent_code=latent_code if not disable_latent_codes else None,
                inHead=inHead,
            )
            target_ray_values = target_s

        coarse_l2, masked_coarse_l2, masked_coarse_l2_weights = mse_loss(mask_target, rgb_coarse[..., :3], target_ray_values[..., :3])
        coarse_cross_entropy, masked_coarse_cross_entropy, masked_coarse_cross_entropy_weights = cross_entropy_loss(mask_target, rgb_coarse[..., 3:], mask_target)
        coarse_loss_mouth = torch.sum(masked_coarse_l2[7:9] + masked_coarse_cross_entropy[7:9])
        coarse_loss = coarse_l2 + 0.02 * coarse_cross_entropy + 0.005 * coarse_loss_mouth

        fine_loss = None
        if rgb_fine is not None:
            fine_l2, masked_fine_l2, masked_fine_l2_weights = mse_loss(mask_target, rgb_fine[..., :3], target_ray_values[..., :3])
            fine_cross_entropy, masked_fine_cross_entropy, masked_fine_cross_entropy_weights = cross_entropy_loss(mask_target, rgb_fine[..., 3:], mask_target)
            fine_loss_mouth = torch.sum(masked_fine_l2[7:9] + masked_fine_cross_entropy[7:9])
            fine_loss = fine_l2 + 0.02 * fine_cross_entropy + 0.005 * fine_loss_mouth
        if dynamic_sampling:
            sample_prob = (masked_coarse_l2_weights + masked_coarse_cross_entropy_weights + masked_fine_l2_weights + masked_fine_cross_entropy_weights) / (
                    masked_coarse_l2_weights.sum() + masked_coarse_cross_entropy_weights.sum() + masked_fine_l2_weights.sum() + masked_fine_cross_entropy_weights.sum())

        loss = 0.0
        latent_code_loss = torch.zeros(1, device=device)
        if train_latent_codes and not disable_latent_codes:
            # latent_code_loss = (torch.norm(latent_code) + torch.norm(latent_code_torso)) * 0.0005
            latent_code_loss = (torch.norm(latent_code)) * 0.0005
        spatial_embeddings_loss = torch.zeros(1, device=device)
        if cfg.models.coarse.use_spatial_embeddings:
            spatial_embeddings_loss = torch.norm(model.spatial_embeddings) * 0.0005

        background_loss = torch.zeros(1, device=device)
        if supervised_train_background:
            background_loss = torch.nn.functional.mse_loss(
                background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
            ).sum(1)
            background_loss = torch.mean(background_loss * weights) * 0.001

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        psnr_rgb = mse2psnr((fine_l2).item())
        loss_mask = fine_cross_entropy

        loss = loss + (latent_code_loss * 10 if regularize_latent_codes else 0.0) \
               + (spatial_embeddings_loss * 10 if regularize_spatial_embedding else 0.0)
        loss_total = loss + (background_loss if supervised_train_background is not None else 0.0)
        loss_total.backward()
        optimizer.step()
        # if cfg.dataset.type.lower() == "audio":
        #     optimizer_Aud.step()
        #     if i >= nosmo_iters:
        #         optimizer_AudAtt.step()
        optimizer.zero_grad()
        # if cfg.dataset.type.lower() == "audio":
        #     optimizer_Aud.zero_grad()
        #     optimizer_AudAtt.zero_grad()
        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
                cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        # for param_group in optimizer_Aud.param_groups:
        #     param_group['lr'] = lr_new
        #
        # for param_group in optimizer_AudAtt.param_groups:
        #     param_group['lr'] = lr_new*5

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " BG Loss: "
                + str(background_loss.item())
                + " PSNR_RGB: "
                + str(psnr_rgb)
                + " LatentReg: "
                + str(latent_code_loss.item())
            )
        # writer.add_scalar("train/loss", loss.item(), i)
        if train_latent_codes:
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
        if train_spatial_embeddings:
            writer.add_scalar("train/spatial_embedding_loss", spatial_embeddings_loss.item(), i)
        if supervised_train_background:
            writer.add_scalar("train/bg_loss", background_loss.item(), i)

        writer.add_scalar("train/coarse_loss_rgb", coarse_l2.item(), i)
        writer.add_scalar("train/coarse_loss_mouth", coarse_loss_mouth.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss_rgb", fine_l2.item(), i)
            writer.add_scalar("train/fine_loss_mouth", fine_loss_mouth.item(), i)
        writer.add_scalar("train/psnr_rgb", psnr_rgb, i)
        writer.add_scalar("train/loss_mask", loss_mask, i)

        # Validation
        if (
                i % cfg.experiment.validate_every == 0
                or i == cfg.experiment.train_iters - 1 and False
        ):
            # torch.cuda.empty_cache()
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, weights = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        driving=driving_target,
                        latent_code=torch.zeros(32, device=device)
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    # Do all validation set...
                    loss, coarse_loss_rgb, fine_loss_rgb, coarse_loss_mask, fine_loss_mask, coarse_loss_mouth_total, fine_loss_mouth_total = 0, 0, 0, 0, 0, 0, 0
                    coarse_loss_mask, fine_loss_mask = 0, 0
                    for img_idx in range(len(validation_data)):
                        img_val, mask_val, pose_val, [H, W, focal], driving_val, _ = validation_data[img_idx]

                        img_target = torch.Tensor(img_val).to(device)
                        mask_target = torch.from_numpy(mask_val).to(device)
                        pose_target = torch.Tensor(pose_val[:3, :4]).to(device)
                        driving_target = torch.Tensor(driving_val).to(device)
                        # if cfg.dataset.type.lower() == "audio":
                        #     driving_target = AudNet(driving_target.unsqueeze(0))

                        # ray_origins, ray_directions = get_ray_bundle_by_mask(
                        #     H, W, focal, pose_target, mask_head
                        # )

                        ray_origins, ray_directions = get_ray_bundle(
                            H, W, focal, pose_target)

                        rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                            H,
                            W,
                            focal,
                            model,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            driving=driving_target,
                            pose=pose_target,
                            pose_c=pose_c,
                            background_prior=background.view(-1, 15) if (train_background or fixed_background) else None,
                            latent_code=None,  # if train_latent_codes or ~disable_latent_codes else None,
                            inHead=mask_target,
                        )
                        # print("did one val")

                        coarse_l2, masked_coarse_l2, masked_coarse_l2_weights = mse_loss(
                            mask_target, rgb_coarse[..., :3], img_target[..., :3])
                        coarse_loss_rgb += coarse_l2
                        coarse_cross_entropy, masked_coarse_cross_entropy, masked_coarse_cross_entropy_weights = cross_entropy_loss(
                            mask_target, rgb_coarse[..., 3:], mask_target)
                        coarse_loss_mask += coarse_cross_entropy
                        coarse_loss_mouth = torch.sum(masked_coarse_l2[7:9] + masked_coarse_cross_entropy[7:9])
                        coarse_loss_mouth_total += coarse_loss_mouth
                        coarse_loss = coarse_l2 + 0.02 * coarse_cross_entropy + 0.005 * coarse_loss_mouth

                        fine_loss = None
                        if rgb_fine is not None:
                            fine_l2, masked_fine_l2, masked_fine_l2_weights = mse_loss(
                                mask_target, rgb_fine[..., :3], img_target[..., :3])
                            fine_loss_rgb += fine_l2
                            fine_cross_entropy, masked_fine_cross_entropy, masked_fine_cross_entropy_weights = cross_entropy_loss(
                                mask_target, rgb_fine[..., 3:], mask_target)
                            fine_loss_mask += fine_cross_entropy
                            fine_loss_mouth = torch.sum(masked_fine_l2[7:9] + masked_fine_cross_entropy[7:9])
                            fine_loss_mouth_total += fine_loss_mouth
                            fine_loss = fine_l2 + 0.02 * fine_cross_entropy + 0.005 * fine_loss_mouth
                        loss = coarse_loss + fine_loss if rgb_fine is not None else None

                    loss /= len(validation_data)
                    psnr_rgb = mse2psnr(fine_loss_rgb.item() / len(validation_data))
                    loss_mask = fine_loss_mask.item() / len(validation_data)
                    writer.add_scalar("validation/loss", loss.item(), i)
                    writer.add_scalar("validation/coarse_loss_rgb", (coarse_loss_rgb.item()) / len(validation_data), i)
                    writer.add_scalar("validation/coarse_loss_mouth", (coarse_loss_mouth.item()) / len(validation_data), i)
                    writer.add_scalar("validation/psnr_rgb", psnr_rgb, i)
                    writer.add_scalar("validation/loss_mask", loss_mask, i)
                    writer.add_image(
                        "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                    )
                    writer.add_image(
                        "validation/mask_coarse", cast_to_image(utils.label2color(rgb_coarse[..., 3:])), i,
                    )
                    if rgb_fine is not None:
                        writer.add_image(
                            "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                        )
                        writer.add_image(
                            "validation/mask_fine", cast_to_image(utils.label2color(rgb_fine[..., 3:])), i,
                        )

                        writer.add_scalar("validation/fine_loss_rgb", (fine_loss_rgb.item()) / len(validation_data), i)
                        writer.add_scalar("validation/fine_loss_mouth", (fine_loss_mouth.item()) / len(validation_data),
                                          i)

                    writer.add_image(
                        "validation/img_target",
                        cast_to_image(img_target[..., :3]),
                        i,
                    )
                    writer.add_image(
                        "validation/mask_target",
                        cast_to_image(utils.label2color(mask_target)),
                        i,
                    )

                    if train_background or fixed_background:
                        writer.add_image(
                            "validation/background", cast_to_image(background[..., :3]), i
                        )
                        writer.add_image(
                            "validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW'
                        )
                    else:
                        writer.add_image(
                            "validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW'
                        )
                    tqdm.write(
                        "Validation loss: "
                        + str(loss.item())
                        + " Validation PSNR_RGB: "
                        + str(psnr_rgb)
                        + " Validation LOSS_MASK: "
                        + str(loss_mask)
                        + " Time: "
                        + str(time.time() - start)
                    )

        # gpu_profile(frame=sys._getframe(), event='line', arg=None)

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "i_batch": i_batch,
                "model_state_dict": model.state_dict(),
                # "aud_state_dict": AudNet.state_dict(),
                # "audattnet_state_dict": AudAttNet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "optimizer_Aud_state_dict": optimizer_Aud.state_dict() if optimizer_Aud is not None else None,
                # "optimizer_AudAtt_state_dict": optimizer_AudAtt.state_dict() if optimizer_AudAtt is not None else None,
                "loss": loss,
                # "psnr": psnr,
                # "lpips": lpips,
                "background": None
                if not (train_background or fixed_background)
                else background.data,
                "latent_codes": None if not train_latent_codes else latent_codes.data,
                "pose_c": pose_c,
                "sample_prob": sample_prob,
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")

# def lpips_single_image_pair_tensor(im1,im2):
#     im1_tensor = im1.unsqueeze(0).permute(2, 0, 1)
#     im2_tensor = im2.unsqueeze(0).permute(2, 0, 1)
#     with torch.no_grad():
#         score = lpips_fn(im1_tensor.reshape((3,-1,32)), im2_tensor.reshape((3,-1,32)))
#         score.cpu()
#     """
#     # TODO check their im loading
#     #img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file)))  RGB image from [-1,1]
#     #img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
#
#     """
#     return score.item()
#
# def lpips_single_image_pair_tensor_2(im1,im2):
#     im1_tensor = im1.permute(2, 0, 1)
#     im2_tensor = im2.permute(2, 0, 1)
#     with torch.no_grad():
#         score = lpips_fn(im1_tensor, im2_tensor)
#         score.cpu()
#     """
#     # TODO check their im loading
#     #img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file)))  RGB image from [-1,1]
#     #img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))
#
#     """
#     return score.item()


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0, 1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


if __name__ == "__main__":
    import signal

    print("before signal registration")
    signal.signal(signal.SIGUSR1, handle_pdb)
    print("after registration")
    # sys.settrace(gpu_profile)

    main()