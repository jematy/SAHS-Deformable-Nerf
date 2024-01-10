import argparse
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import subprocess
# proc1 = subprocess.Popen(['scontrol', 'show', 'job', os.environ['SLURM_JOBID'], '-d'], stdout=subprocess.PIPE)
# process = subprocess.run(['grep', '-oP', 'GRES=.*IDX:\K\d'], stdin=proc1.stdout, capture_output=True, text=True)
# os.environ['EGL_DEVICE_ID'] = process.stdout.rstrip()
# proc1.stdout.close()


import imageio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# matplotlib.use("TkAgg")

import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm
# from nerf-pytorch import


from nerf import (
    CfgNode,
    get_ray_bundle,
    get_ray_bundle_by_mask,
    load_flame_data,
    load_llff_data,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
    meshgrid_xy,
    utils
)


def unproject_torch(depth, focal, world_coords=True, pose=None, name=None, H=512, W=512, save=True):
    raster = meshgrid_xy(torch.arange(W), torch.arange(H))
    u = raster[0]
    v = raster[1]
    x = (u - focal[2]) * depth.cpu() / ((focal[0]))
    y = (v - focal[3]) * depth.cpu() / ((focal[1]))
    z = depth.cpu()
    pts = torch.dstack((x, y, z)).reshape(-1, 3)
    dims = 3
    if world_coords and pose is not None:
        points_hom = torch.cat((pts, torch.ones(pts.shape[0], 1)), dim=1)  # .to(depth.device)
        pts = torch.matmul(torch.inverse(pose).cpu(), points_hom.transpose(1, 0)).transpose(0, 1)
        dims = 4
    save_pc(pts, name, dims=dims)
    return pts


def save_pc(tensor, name=None, dims=4):
    from pathlib import Path
    if name is None:
        from datetime import datetime
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%Y_%H:%M:%S")

    fname = name if name is not None else date_time + ".obj"

    #     print("saving point cloud %s" % fname)

    Path(fname).write_text("\n".join([f"v {p[0]} {p[1]} {p[2]}" for p in tensor.reshape(-1, dims)]))
    return


def save_plt_image(im1, outname):
    fig = plt.figure()
    fig.set_size_inches((6.4, 6.4))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.set_cmap('jet')
    ax.imshow(im1, aspect='equal')
    plt.savefig(outname, dpi=80)
    plt.close(fig)


def normal_map_from_depth_map(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            n = np.array([-dzdx, -dzdy, 0.005])

            n = n * 1 / np.linalg.norm(n)
            dir = np.array([x, y, 1.0])
            dir = dir * 1 / np.linalg.norm(dir)

            normals[x, y] = (n * 0.5 + 0.5)
            phong[x, y] = np.dot(dir, n) * 0.5 + 0.5

    normals *= 255
    normals = normals.astype('uint8')
    # plt.imshow(depthmap, cmap='gray')
    # plt.show()
    plt.imshow(normals)
    plt.show()
    plt.imshow(phong)
    plt.show()
    print('a')
    return normals


def torch_normal_map(depthmap, focal, weights=None, clean=True, central_difference=False):
    W, H = depthmap.shape
    # normals = torch.zeros((H,W,3), device=depthmap.device)
    cx = focal[2] * W
    cy = focal[3] * H
    fx = focal[0]
    fy = focal[1]
    ii, jj = meshgrid_xy(torch.arange(W, device=depthmap.device),
                         torch.arange(H, device=depthmap.device))
    points = torch.stack(
        [
            ((ii - cx) * depthmap) / fx,
            -((jj - cy) * depthmap) / fy,
            depthmap,
        ],
        dim=-1)
    difference = 2 if central_difference else 1
    dx = (points[difference:, :, :] - points[:-difference, :, :])
    dy = (points[:, difference:, :] - points[:, :-difference, :])
    normals = torch.cross(dy[:-difference, :, :], dx[:, :-difference, :], 2)
    normalize_factor = torch.sqrt(torch.sum(normals * normals, 2))
    normals[:, :, 0] /= normalize_factor
    normals[:, :, 1] /= normalize_factor
    normals[:, :, 2] /= normalize_factor
    normals = normals * 0.5 + 0.5

    if clean and weights is not None:  # Use volumetric rendering weights to clean up the normal map
        mask = weights.repeat(3, 1, 1).permute(1, 2, 0)
        mask = mask[:-difference, :-difference]
        where = torch.where(mask > 0.22)
        normals[where] = 1.0
        normals = (1 - mask) * normals + (mask) * torch.ones_like(normals)
    normals *= 255
    # plt.imshow(normals.cpu().numpy().astype('uint8'))
    # plt.show()
    return normals


def vis(tensor):
    plt.imshow((tensor * 255).cpu().numpy().astype('uint8'))
    plt.show()


def normal_map_from_depth_map_backproject(depthmap):
    h, w = np.shape(depthmap)
    normals = np.zeros((h, w, 3))
    phong = np.zeros((h, w, 3))
    cx = cy = h // 2
    fx = fy = 500
    fx = fy = 1150
    for x in range(1, h - 1):
        for y in range(1, w - 1):
            # dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
            # dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

            p = np.array([(x * depthmap[x, y] - cx) / fx, (y * depthmap[x, y] - cy) / fy, depthmap[x, y]])
            py = np.array(
                [(x * depthmap[x, y + 1] - cx) / fx, ((y + 1) * depthmap[x, y + 1] - cy) / fy, depthmap[x, y + 1]])
            px = np.array(
                [((x + 1) * depthmap[x + 1, y] - cx) / fx, (y * depthmap[x + 1, y] - cy) / fy, depthmap[x + 1, y]])

            # n = np.array([-dzdx, -dzdy, 0.005])
            n = np.cross(px - p, py - p)
            n = n * 1 / np.linalg.norm(n)
            dir = p  # np.array([x,y,1.0])
            dir = dir * 1 / np.linalg.norm(dir)

            normals[x, y] = (n * 0.5 + 0.5)
            phong[x, y] = np.dot(dir, n) * 0.5 + 0.5

    normals *= 255
    normals = normals.astype('uint8')
    # plt.imshow(depthmap, cmap='gray')
    # plt.show()
    # plt.imshow(normals)
    # plt.show()
    # plt.imshow(phong)
    # plt.show()
    # print('a')
    return normals


def error_image(im1, im2):
    fig = plt.figure()
    diff = (im1 - im2)
    # gt_vs_theirs[total_mask, :] = 0
    # print("theirs ", np.sqrt(np.sum(np.square(gt_vs_theirs))), np.mean(np.square(gt_vs_theirs)))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
    # they are still used in the computation of the image padding.
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
    # by default they leave padding between the plotted data and the frame. We use tigher=True
    # to make sure the data gets scaled to the full extents of the axes.
    plt.autoscale(tight=True)
    plt.imshow(np.linalg.norm(diff, axis=2), cmap='jet')
    # ax.plt.axes('off')

    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # plt.show()
    return fig


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0, 1.0)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--savedir", type=str, default='./renders/', help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-masks-image", action="store_true", help="Save disparity images too."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    parser.add_argument(
        "--save-error-image", action="store_true", help="Save photometric error visualization"
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "expression":
        # Load blender dataset

        from nerf import nerface_dataloader
        test_data = nerface_dataloader.NerfaceDataset(
            mode='val',
            cfg=cfg,
            # N_max=100
        )

    if cfg.dataset.type.lower() == "audio":
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from nerf import audio_dataloader
        test_data = audio_dataloader.AudioDataset(
            mode='val',
            cfg=cfg
        )

    H, W = test_data.H, test_data.W
    focal = 0
    hwf = [H, W, focal]

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    model = getattr(models, cfg.models.mask.type)(cfg)
    model.to(device)

    checkpoint = torch.load(configargs.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]
    if "background" in checkpoint.keys():
        background = checkpoint["background"]
        if background is not None:
            print("loaded background with shape ", background.shape)
            background.to(device)
    if "latent_codes" in checkpoint.keys():
        latent_codes = checkpoint["latent_codes"]
        use_latent_code = False
        if latent_codes is not None:
            use_latent_code = True
            latent_codes.to(device)
            print("loading index map for latent codes...")
            idx_map = np.load(cfg.dataset.basedir + "/index_map.npy").astype(int)
            print("loaded latent codes from checkpoint, with shape ", latent_codes.shape)
    pose_c = None
    if "pose_c" in checkpoint.keys():
        pose_c = checkpoint["pose_c"]

    model.eval()

    replace_background = False
    if replace_background:
        from PIL import Image
        # background = Image.open('./view.png')
        background = Image.open(cfg.dataset.basedir + '/bg/00050.png')
        # background = Image.open("./real_data/andrei_dvp/" + '/bg/00050.png')
        background.thumbnail((H, W))
        background = torch.from_numpy(np.array(background).astype(float)).to(device)
        background = background[..., :3] / 255
        print('loaded custom background of shape', background.shape)

        # background = torch.ones_like(background)
        # background.permute(2,0,1)

    # render_poses = render_poses.float().to(device)

    # Create directory to save images to.
    os.makedirs(configargs.savedir, exist_ok=True)
    os.makedirs(os.path.join(configargs.savedir, "masks"), exist_ok=True)
    if configargs.save_disparity_image:
        os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)
    if configargs.save_error_image:
        os.makedirs(os.path.join(configargs.savedir, "error"), exist_ok=True)
    os.makedirs(os.path.join(configargs.savedir, "normals"), exist_ok=True)
    os.makedirs(os.path.join(configargs.savedir, "mesh"), exist_ok=True)
    # Evaluation loop
    times_per_image = []

    # render_poses = render_poses.float().to(device)
    # render_poses = poses[i_test].float().to(device)
    # expressions = torch.arange(-6,6,0.5).float().to(device)
    # render_expressions = expressions[i_test].float().to(device)
    # avg_img = torch.mean(images[i_train],axis=0)
    # avg_img = torch.ones_like(avg_img)

    # pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # for i, pose in enumerate(tqdm(render_poses)):
    index_of_image_after_train_shuffle = 0
    # render_expressions = render_expressions[[300]] ### TODO render specific expression
    render_expressions = None
    #######################
    no_background = False
    no_expressions = False
    no_lcode = True
    nerf = False
    frontalize = False
    interpolate_mouth = False

    #######################
    if nerf:
        no_background = True
        no_expressions = True
        no_lcode = True
    if no_background: background = None
    if no_expressions: render_expressions = torch.zeros_like(render_expressions, device=render_expressions.device)
    # if no_lcode:
    #     use_latent_code = True
    #     latent_codes = torch.zeros(5000, 32, device=device)

    for i in tqdm(range(len(test_data))):
        if i >= 1500:
            break
        _, mask, pose, hwf, driving, _, fname = test_data[i]
        H, W, focal = hwf
        # if i%25 != 0: ### TODO generate only every 25th im
        # if i != 511: ### TODO generate only every 25th im
        #    continue
        # if i<4500:
        #    continue
        start = time.time()
        rgb = None, None
        disp = None, None
        with torch.no_grad():
            # pose = render_poses[i]

            mask_target = torch.from_numpy(utils.shrink(mask)).to(device)
            mask_target_label = mask_target
            if interpolate_mouth:
                frame_id = 241
                num_images = 150
                pose = render_poses[241]
                expression = render_expressions[241].clone()
                expression[68] = torch.arange(-1, 1, 2 / 150, device=device)[i]

            if frontalize:
                pose = render_poses[0]
            # pose = render_poses[300] ### TODO fixes pose
            # expression = render_expressions[0] ### TODO fixes expr
            # expression = torch.zeros_like(expression).to(device)

            # ablate = 'view_dir'
            ablate = None

            if ablate == 'expression':
                pose = render_poses[100]
            elif ablate == 'latent_code':
                pose = render_poses[100]
                expression = render_expressions[100]
                if idx_map[100 + i, 1] >= 0:
                    # print("found latent code for this image")
                    index_of_image_after_train_shuffle = idx_map[100 + i, 1]
            elif ablate == 'view_dir':
                pose = render_poses[100]
                expression = render_expressions[100]
                _, ray_directions_ablation = get_ray_bundle(hwf[0], hwf[1], hwf[2], render_poses[240 + i][:3, :4])

            pose = torch.Tensor(pose[:3, :4]).to(device)
            driving = torch.Tensor(driving).to(device)

            # pose = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
            if use_latent_code:
                # pass
                # if idx_map[i,1] >= 0:
                #    #print("found latent code for this image")
                #    index_of_image_after_train_shuffle = idx_map[i,1]
                # index_of_image_after_train_shuffle = 10 ## TODO Fixes latent code
                # index_of_image_after_train_shuffle = idx_map[84,1] ## TODO Fixes latent code v2 for andrei
                index_of_image_after_train_shuffle = idx_map[
                    10, 1]  ## TODO Fixes latent code - USE THIS if not ablating!

                latent_code = latent_codes[index_of_image_after_train_shuffle].to(device) if use_latent_code else None

            # latent_code = torch.mean(latent_codes)
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            # ray_origins, ray_directions = get_ray_bundle_by_mask(
            #     H, W, focal, pose, mask_head
            # )
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _, weights, depth_fine = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                driving=driving,
                pose=pose,
                pose_c=pose_c,
                background_prior=background.view(-1, 15) if (background is not None) else None,
                # background_prior = torch.ones_like(background).view(-1,3),  # White background
                latent_code=None,
                inHead=mask_target,
                # ray_directions_ablation = ray_directions_ablation
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
            normals = torch_normal_map(disp_fine, focal, weights, clean=True)
            # normals = normal_map_from_depth_map_backproject(disp_fine.cpu().numpy())
            ## TODO uncomment to save normal maps:
            save_plt_image(normals.cpu().numpy().astype('uint8'),
                           os.path.join(configargs.savedir, 'normals', f"{i:04d}.png"))
            # if configargs.save_disparity_image:
            if False:
                disp = disp_fine if disp_fine is not None else disp_coarse
                # normals = normal_map_from_depth_map_backproject(disp.cpu().numpy())
                normals = normal_map_from_depth_map_backproject(disp_fine.cpu().numpy())
                save_plt_image(normals.astype('uint8'), os.path.join(configargs.savedir, 'normals', f"{i:04d}.png"))

            # if configargs.save_normal_image:
            #    normal_map_from_depth_map_backproject(disp_fine.cpu().numpy())
        # rgb[torch.where(weights>0.25)]=1.0
        # rgb[torch.where(weights>0.1)] = (rgb * weights + (torch.ones_like(weights)-weights)*torch.ones_like(weights))
        times_per_image.append(time.time() - start)

        if configargs.savedir and cfg.dataset.type.lower() == "expression":
            ## for MPI comparison:
            from math import floor
            dir = configargs.savedir  # + ('/%d' % int(floor(i*12/(len(test_data)))))
            os.makedirs(dir, exist_ok=True)
            savefile = os.path.join(dir, f"f_{i:04d}.png")
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3])
            )
            savesegfile = os.path.join(dir, 'masks', f"f_{i:04d}.png")
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            seg = utils.label2color(rgb[..., 3:])
            imageio.imwrite(
                savesegfile, cast_to_image(seg)
            )
            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.savedir, "disparity", f"f_{i:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp_fine))
            if configargs.save_error_image:
                savefile = os.path.join(configargs.savedir, "error", f"f_{i:04d}.png")
                GT = images[i_test][i]
                fig = error_image(GT, rgb.cpu().numpy())
                # imageio.imwrite(savefile, cast_to_disparity_image(disp))
                plt.savefig(savefile, pad_inches=0, bbox_inches='tight', dpi=54)

            # savefile = os.path.join(dir, 'mesh', f"f_{i:04d}.obj")
            # unproject_torch(depth_fine, hwf[2], H=hwf[0], W=hwf[1], name=savefile)

        # tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")

        if configargs.savedir and cfg.dataset.type.lower() == "audio":
            ## for MPI comparison:
            from math import floor
            dir = configargs.savedir  # + ('/%d' % int(floor(i*12/(len(test_data)))))
            os.makedirs(dir, exist_ok=True)
            fname = fname.split('/')[-1]
            savefile = os.path.join(dir, fname)
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3])
            )
            savesegfile = os.path.join(dir, 'masks', fname.split('.')[0] + '.png')
            # savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            seg = utils.label2color(rgb[..., 3:])
            imageio.imwrite(
                savesegfile, cast_to_image(seg)
            )
            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.savedir, "disparity", fname.split('.')[0] + '.png')
                imageio.imwrite(savefile, cast_to_disparity_image(disp_fine))
            if configargs.save_error_image:
                savefile = os.path.join(configargs.savedir, "error", fname.split('.')[0] + '.png')
                GT = images[i_test][i]
                fig = error_image(GT, rgb.cpu().numpy())
                # imageio.imwrite(savefile, cast_to_disparity_image(disp))
                plt.savefig(savefile, pad_inches=0, bbox_inches='tight', dpi=54)
            # savefile = os.path.join(dir, 'mesh', fname.split('.')[0] + '.obj')
            # unproject_torch(depth_fine, hwf[2], H=hwf[0], W=hwf[1], name=savefile)
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")


if __name__ == "__main__":
    main()
