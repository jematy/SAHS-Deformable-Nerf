import cv2
import imageio
import torch
from torch.utils import data
import json
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from tqdm import tqdm

class Spade_NerfaceDataset(Dataset):
    def __init__(self, mode, cfg, debug=False):
        self.cfg = cfg
        print("initializing NerfaceDataset with mode %s" % mode)
        self.mode = mode
        basedir = cfg.dataset.basedir
        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.metas = json.load(fp)
        frame = self.metas["frames"][0]
        fname = os.path.join(basedir, "./"+ self.mode + "/" + frame["file_path"] + ".png")
        #这里的self.mode是train，导致找不到文件，不改这里的话，就把文件放到train文件夹下，也就是train文件夹里面还有一个train文件夹
        # fname = os.path.join(basedir, "./" + frame["file_path"] + ".png")
        im = imageio.imread(fname)
        self.H, self.W = im.shape[:2]
        fnames = []
        for i, frame in enumerate(tqdm(self.metas["frames"])):
            fname = os.path.join(basedir, "./"+ self.mode + "/" + frame["file_path"] + ".png")
            fnames.append(fname)
        print("Done with data loading")

        self.fnames = fnames

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.read_image(idx)
        elif isinstance(idx, slice):

            start = idx.start
            stop = idx.stop
            step = idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.__len__()
            if step is None:
                step = 1

            imgs = []
            fnames = []
            for i in range(start, stop, step):
                img, fname = self.read_image(i)
                imgs.append(img)
                fnames.append(fname)
            return imgs, fnames

    def __len__(self):
        return len(self.fnames)

    def read_image(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(fname), code=cv2.COLOR_BGR2RGB)
        img = (np.array(img) / 255.0).astype(np.float32)
        img = cv2.resize(img, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
        if self.cfg.nerf.train.white_background:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
        return img, fname


class Spade_NerfaceDataset_output(Dataset):
    def __init__(self, mode, cfg, debug=False):
        self.cfg = cfg
        print("initializing NerfaceDataset with mode %s" % mode)
        self.mode = mode
        basedir = cfg.dataset.basedir
        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.metas = json.load(fp)
        frame = self.metas["frames"][0]
        # for i in range(5507):
        #     if i == 437:
        #         continue
        #     file_paths = [f"f_{i:04}" ]
        file_paths = [f"f_{i:04}" for i in range(5435)]
        self.metas["frames"] = frame['file_path'] = file_paths
        fname = os.path.join("/home/user/jkx/nerf-pytorch/renders/person_3_rendered_frames_texture_not_half_res_5Ktrain"+ "/" + frame["file_path"][0] + ".png")
        #这里的self.mode是train，导致找不到文件，不改这里的话，就把文件放到train文件夹下，也就是train文件夹里面还有一个train文件夹
        # fname = os.path.join(basedir, "./" + frame["file_path"] + ".png")
        im = imageio.imread(fname)
        self.H, self.W = im.shape[:2]
        fnames = []
        for i, frame in enumerate(tqdm(self.metas["frames"])):
            # print("/data/jkx/projects/nerf-pytorch/renders/person_1_rendered_frames/" + frame["file_path"] + ".png")
            # e = "/data/jkx/projects/nerf-pytorch/renders/person_1_rendered_frames/" + frame + ".png"
            fname = os.path.join("/home/user/jkx/nerf-pytorch/renders/person_3_rendered_frames_texture_not_half_res_5Ktrain/" + frame + ".png")
            fnames.append(fname)
        print("Done with data loading")

        self.fnames = fnames

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.read_image(idx)
        elif isinstance(idx, slice):

            start = idx.start
            stop = idx.stop
            step = idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.__len__()
            if step is None:
                step = 1

            imgs = []
            fnames = []
            for i in range(start, stop, step):
                img, fname = self.read_image(i)
                imgs.append(img)
                fnames.append(fname)
            return imgs, fnames

    def __len__(self):
        return len(self.fnames)

    def read_image(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(fname), code=cv2.COLOR_BGR2RGB)
        img = (np.array(img)).astype(np.float32)
        img = cv2.resize(img, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
        if self.cfg.nerf.train.white_background:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
        return img, fname


# def get_texture_identy_photo(mode, cfg):
#     basedir = cfg.dataset.basedir
#     fname = os.path.join(basedir, "./" + mode + "/" + "./identity_photo/f_0437" + ".png")
#     print("Done with data loading")
#     img = cv2.cvtColor(cv2.imread(fname), code=cv2.COLOR_BGR2RGB)
#     img = (np.array(img) / 255.0).astype(np.float32)
#     H, W = img.shape[:2]
#     img = cv2.resize(img, dsize=(H, W), interpolation=cv2.INTER_AREA)
#     if cfg.nerf.train.white_background:
#         img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
#     return img, fname
