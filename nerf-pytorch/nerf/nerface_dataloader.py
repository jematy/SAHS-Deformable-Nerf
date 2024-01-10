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


class NerfaceDataset(Dataset):
    def __init__(self, mode, cfg, debug=False):
        self.cfg = cfg
        print("initializing NerfaceDataset with mode %s" % mode)
        self.mode = mode
        basedir = cfg.dataset.basedir
        load_bbox = True
        self.debug = debug
        self.load_segmaps = cfg.models.mask.use_mask

        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.metas = json.load(fp)

        # get size
        frame = self.metas["frames"][0]
        fname = os.path.join(basedir, "./"+ self.mode + "/" + frame["file_path"] + ".png")
        #这里的self.mode是train，导致找不到文件，不改这里的话，就把文件放到train文件夹下，也就是train文件夹里面还有一个train文件夹
        # fname = os.path.join(basedir, "./" + frame["file_path"] + ".png")
        im = imageio.imread(fname)
        self.H, self.W = im.shape[:2]

        camera_angle_x = float(self.metas["camera_angle_x"])
        focal = 0.5 * self.W / np.tan(0.5 * camera_angle_x)

        # focals = (meta["focals"])
        # intrinsics = self.metas["intrinsics"] if self.metas["intrinsics"] else None
        if self.metas["intrinsics"]:
            self.intrinsics = np.array(self.metas["intrinsics"])
        else:
            self.intrinsics = np.array([focal, focal, 0.5, 0.5])  # fx fy cx cy

        # In debug mode, return extremely tiny images
        if debug:
            self.H = self.H // 32
            self.W = self.W // 32
            # focal = focal / 32.0
            self.intrinsics[:2] = self.intrinsics[:2] / 32.0

        if self.cfg.dataset.half_res:
            self.H = self.H // 2
            self.W = self.W // 2
            # focal = focal / 2.0
            self.intrinsics[:2] = self.intrinsics[:2] * 0.5

        poses = []
        expressions = []
        bboxs = []
        fnames = []
        segnames = []
        # Prepare importance sampling maps
        # probs = np.zeros((len(self.metas["frames"]), self.H, self.W))
        # p = 0.9
        # probs.fill(1 - p)
        # print("computing bounding boxes probability maps")
        for i, frame in enumerate(tqdm(self.metas["frames"])):
            fname = os.path.join(basedir, "./"+ self.mode + "/" + frame["file_path"] + ".png")
            fnames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))

            if self.load_segmaps:
                segname = os.path.join(basedir, "./" + self.mode + "/masks/" + frame["file_path"] + ".png")
                segnames.append(segname)

            if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0, 1.0, 0.0, 1.0]))
                else:
                    # if mode == 'train':
                    bbox = np.array(frame["bbox"])
                    bbox[0:2] *= self.H
                    bbox[2:4] *= self.W
                    bbox = np.floor(bbox).astype(int)
                    # probs[i, bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
                    # probs[i] = (1 / probs[i].sum()) * probs[i]
                    bboxs.append(bbox)

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.int32)

        #counts.append(counts[-1] + imgs.shape[0])
        #all_imgs.append(imgs)
        #all_frontal_imgs.append(frontal_imgs)
        #all_poses.append(poses)
        #all_expressions.append(expressions)
        #all_bboxs.append(bboxs)

        #i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

        #imgs = np.concatenate(all_imgs, 0)

        #poses = np.concatenate(all_poses, 0)
        #expressions = np.concatenate(all_expressions, 0)
        #bboxs = np.concatenate(all_bboxs, 0)


        # if type(focals) is list:
        #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
        # else:
        #     focal = np.array([focal, focal])

        # render_poses = torch.stack(
        #     [
        #         torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
        #         for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        #     ],
        #     0,
        # )

        print("Done with data loading")

        self.bboxs = bboxs
        self.poses = poses
        self.expressions = expressions
        self.fnames = fnames
        self.segnames = segnames
        # self.probs = probs

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

            poses = []
            expressions = []
            imgs = []
            segs = []
            bboxs = []
            hwks = []
            fnames = []
            for i in range(start, stop, step):
                img, seg, pose, hwk, expression, bbox, fname = self.read_image(i)
                imgs.append(img)
                if seg is not None:
                    segs.append(seg)
                poses.append(pose)
                hwks.append(hwk)
                expressions.append(expression)
                # probs.append(prob)
                fnames.append(fname)
            return imgs, segs, poses, hwks, expressions, bboxs, fnames

    def __len__(self):
        return self.poses.shape[0]

    def read_image(self, idx):
        pose = self.poses[idx]
        expression = self.expressions[idx]
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(fname), code=cv2.COLOR_BGR2RGB)
        img = (np.array(img) / 255.0).astype(np.float32)
        img = cv2.resize(img, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
        if self.cfg.nerf.train.white_background:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])#将一个带有透明度的图像合成到一个白色的背景上
        seg = None
        if self.load_segmaps:
            segname = self.segnames[idx]
            seg = cv2.imread(segname)
            from . import utils
            seg = np.array(seg)
            seg = utils.color2label_np(seg).astype(np.float32)
            seg = cv2.resize(seg, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
        return img, seg, pose, [self.H, self.W, self.intrinsics], expression, self.bboxs[idx], fname
