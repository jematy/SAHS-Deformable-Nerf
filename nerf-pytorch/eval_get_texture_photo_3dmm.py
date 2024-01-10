import argparse
import os
import time

# import subprocess
# proc1 = subprocess.Popen(['scontrol', 'show', 'job', os.environ['SLURM_JOBID'], '-d'], stdout=subprocess.PIPE)
# process = subprocess.run(['grep', '-oP', 'GRES=.*IDX:\K\d'], stdin=proc1.stdout, capture_output=True, text=True)
# os.environ['EGL_DEVICE_ID'] = process.stdout.rstrip()
# proc1.stdout.close()


import imageio
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import torch
import torchvision
import yaml
from tqdm import tqdm
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
from nerf import (CfgNode)
import numpy as np
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
    parser = argparse.ArgumentParser()
    # print(parser)
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
    cfg = None

    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "expression":
        # Load blender dataset

        from torch.utils.data import Dataset
        from nerf import texture_loader
        test_data = texture_loader.Spade_NerfaceDataset_output(
            mode='test',
            cfg=cfg,
            # N_max=100
        )
    if cfg.dataset.type.lower() == "audio":
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from nerf import texture_loader
        test_data = texture_loader.Spade_AudioDataset_output(
            mode='val',
            cfg=cfg
        )
        from nerf import audio_dataloader
        audio_val_data = audio_dataloader.AudioDataset(
            mode='val',
            cfg=cfg
        )
        # test_data = texture_loader.Spade_AudioDataset_output(
        #     mode='train',
        #     cfg=cfg
        # )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    G = Generator().to(device)

    checkpoint = torch.load(configargs.checkpoint)
    G.load_state_dict(checkpoint["model_state_dict"])

    G.eval()

    if True:
        indenty_photo, frame_identy_photo = get_texture_identy_photo(
            mode='test',
            cfg=cfg
        )

    for i in tqdm(range(len(test_data)), desc="Test", unit="photo"):
        image, _ = test_data[i]
        frame = indenty_photo
        frame = torch.from_numpy(frame).to(device)
        frame = frame.unsqueeze(0)
        frame = frame.permute(0, 3, 1, 2)
        image = torch.from_numpy(image).to(device)
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        with torch.no_grad():
            generated_photo = G(frame, image)
            generated_photo = generated_photo.permute(0, 2, 3, 1)
            generated_photo = generated_photo.cpu().numpy()
            generated_photo = generated_photo.squeeze(0)
            if True:#to avoid negative values
                generated_photo = np.clip(generated_photo, 0, 1)
            generated_photo = generated_photo * 255
            generated_photo = generated_photo.astype(np.uint8)
            im_pil = Image.fromarray(generated_photo)
            dir = configargs.savedir  # + ('/%d' % int(floor(i*12/(len(test_data)))))
            if not os.path.exists(os.path.join(dir)):
                os.makedirs(os.path.join(dir))
            savefile = os.path.join(dir, f"f_{i:04d}.png")
            im_pil.save(savefile)
            # imageio.imwrite(
            #     savefile, generated_photo
            # )
            print("Image saved successfully!")


if __name__ == "__main__":
    main()
