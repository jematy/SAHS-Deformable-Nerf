import torch
import numpy as np


def shrink(mask):
    mask = np.argmax(mask, axis=-1)
    COLOR_MAP = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    shrink_mask = np.zeros((mask.shape[0], mask.shape[1], 12), dtype=np.int32)
    for key in COLOR_MAP:
        shrink_mask[mask == key] = COLOR_MAP[key]
    return shrink_mask


def color2label_np(target):
    mask = np.zeros((target.shape[0], target.shape[1], 12), dtype=np.int32)
    maps = np.array(
        [
            [0, 0, 0],  # 背景
            [204, 0, 0],  # 脸
            [76, 153, 0],  # 鼻子
            [204, 204, 0],  # 眼镜
            [51, 51, 255],  # 眼
            [0, 255, 255],  # 眉毛
            [102, 51, 0],  # 耳朵
            [102, 204, 0],  # 嘴巴内部
            [255, 255, 0],  # 嘴唇
            [0, 0, 204],  # 头发
            [255, 153, 51],  # 脖子
            [0, 204, 0]  # 躯干
        ],
        dtype=np.int32
    )
    maps_one = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.int32
    )
    for map, map_one in zip(maps, maps_one):
        index = np.where(np.all(target[:, :] == map, axis=-1))
        mask[index[0], index[1], :] = map_one
    return mask


# def color2label(target):
#     target = target.cpu()
#     mask = torch.zeros((target.shape[0], target.shape[1], 12), dtype=torch.int32)
#     maps = torch.tensor(
#         [
#             [0, 0, 0],  # 背景
#             [204, 0, 0],  # 脸
#             [76, 153, 0],  # 鼻子
#             [204, 204, 0],  # 眼镜
#             [51, 51, 255],  # 眼
#             [0, 255, 255],  # 眉毛
#             [102, 51, 0],  # 耳朵
#             [102, 204, 0],  # 嘴巴内部
#             [255, 255, 0],  # 嘴唇
#             [0, 0, 204],  # 头发
#             [255, 153, 51],  # 脖子
#             [0, 204, 0]  # 躯干
#         ],
#         dtype=torch.int
#     )
#     maps_one = torch.tensor(
#         [
#             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#         ],
#         dtype=torch.int
#     )
#     for map, map_one in zip(maps, maps_one):
#         index = torch.where((target[:, :] == map).sum(dim=-1) == 3)
#         mask[index[0], index[1], :] = map_one
#     return mask


def label2color(mask):
    mask = torch.argmax(mask, dim=-1)
    # COLOR_MAP = {
    #     0: [0, 0, 0],
    #     1: [1, 0, 0],
    #     2: [0, 1, 0],
    #     3: [1, 0, 1],
    #     4: [1, 1, 0],
    #     5: [0, 0, 1]
    # }
    COLOR_MAP = {
        0: [0, 0, 0],  # 背景
        1: [204, 0, 0],  # 脸
        2: [76, 153, 0],  # 鼻子
        3: [204, 204, 0],  # 眼镜
        4: [51, 51, 255],  # 眼
        5: [0, 255, 255],  # 眉毛
        6: [102, 51, 0],  # 耳朵
        7: [102, 204, 0],  # 嘴巴内部
        8: [255, 255, 0],  # 嘴唇
        9: [0, 0, 204],  # 头发
        10: [255, 153, 51],  # 脖子
        11: [0, 204, 0]  # 躯干
    }
    color = torch.zeros((mask.shape[0], mask.shape[1], 3), dtype=torch.float32)
    for key in COLOR_MAP:
        color[mask == key] = torch.tensor(COLOR_MAP[key][::-1], dtype=torch.float32)
    color = color / 255.
    return color
