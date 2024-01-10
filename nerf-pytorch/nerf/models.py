import numpy as np
import torch
from . import modules
from .nerf_helpers import get_embedding_function


class VeryTinyNeRFModel(torch.nn.Module):
    r"""Define a "very tiny" NeRF model comprising three fully connected layers.
    """

    def __init__(self, filter_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(VeryTinyNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 65 -> 128)
        self.layer1 = torch.nn.Linear(
            self.xyz_encoding_dims + self.viewdir_encoding_dims, filter_size
        )
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MultiHeadNeRFModel(torch.nn.Module):
    r"""Define a "multi-head" NeRF model (radiance and RGB colors are predicted by
    separate heads).
    """

    def __init__(self, hidden_size=128, num_encoding_functions=6, use_viewdirs=True):
        super(MultiHeadNeRFModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        if use_viewdirs is True:
            self.viewdir_encoding_dims = 3 + 3 * 2 * num_encoding_functions
        else:
            self.viewdir_encoding_dims = 0
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(self.xyz_encoding_dims, hidden_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 3_1 (default: 128 -> 1): Predicts radiance ("sigma")
        self.layer3_1 = torch.nn.Linear(hidden_size, 1)
        # Layer 3_2 (default: 128 -> 1): Predicts a feature vector (used for color)
        self.layer3_2 = torch.nn.Linear(hidden_size, hidden_size)

        # Layer 4 (default: 39 + 128 -> 128)
        self.layer4 = torch.nn.Linear(
            self.viewdir_encoding_dims + hidden_size, hidden_size
        )
        # Layer 5 (default: 128 -> 128)
        self.layer5 = torch.nn.Linear(hidden_size, hidden_size)
        # Layer 6 (default: 128 -> 3): Predicts RGB color
        self.layer6 = torch.nn.Linear(hidden_size, 3)

        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x, view = x[..., : self.xyz_encoding_dims], x[..., self.xyz_encoding_dims:]
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        sigma = self.layer3_1(x)
        feat = self.relu(self.layer3_2(x))
        x = torch.cat((feat, view), dim=-1)
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return torch.cat((x, sigma), dim=-1)


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
            self,
            hidden_size=256,
            num_layers=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            **kwargs
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        return torch.cat((rgb, alpha), dim=-1)


class PaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
    ):
        super(PaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x = xyz  # self.relu(self.layers_xyz[0](xyz))
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)

class NeRFaceModel(torch.nn.Module):
    def __init__(self, cfg):
        super(NeRFaceModel, self).__init__()
        self.num_coarse = cfg.nerf.train.num_coarse
        self.num_fine = cfg.nerf.train.num_fine

        self.use_spatial_embeddings = cfg.models.coarse.use_spatial_embeddings
        self.use_viewdirs = cfg.models.coarse.use_viewdirs
        self.use_ambient = cfg.models.hyper.use_ambient

        self.spatial_embeddings = None
        if self.use_spatial_embeddings:
            self.spatial_embeddings = torch.nn.Parameter(torch.randn(1, 32, 32, 32, 32) * 0.01)

        self.encode_pose_fn = get_embedding_function(
            num_encoding_functions=3,
            include_input=False,
            log_sampling=True,
        )

        self.encode_position_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
            include_input=cfg.models.coarse.include_input_xyz,
            log_sampling=cfg.models.coarse.log_sampling_xyz,
        )

        self.encode_direction_fn = None
        if self.use_viewdirs:
            self.encode_direction_fn = get_embedding_function(
                num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
                include_input=cfg.models.coarse.include_input_dir,
                log_sampling=cfg.models.coarse.log_sampling_dir,
            )

        self.encode_ambient_fn = None
        if self.use_ambient:
            self.encode_ambient_fn = get_embedding_function(
                num_encoding_functions=cfg.models.hyper.num_encoding_fn_ambient,
                include_input=cfg.models.hyper.include_input_ambient,
                log_sampling=cfg.models.hyper.log_sampling_ambient,
            )

        self.use_warp = cfg.models.warp.use_warp
        if self.use_warp:
            self.warp_field_mlp = getattr(modules, cfg.models.warp.type)(
                num_layers=cfg.models.warp.num_layers,
                hidden_size=cfg.models.warp.hidden_size,
                skip_connect_every=cfg.models.warp.skip_connect_every,
                num_encoding_fn_xyz=cfg.models.warp.num_encoding_fn_xyz,
                include_driving=True,
                include_pose=False,
            )

        self.hyper_include_driving = cfg.models.hyper.include_driving
        self.hyper_slice_method = cfg.models.hyper.slice_method
        if self.use_ambient:
            if self.hyper_slice_method == 'bendy_sheet':
                self.hyper_sheep_mlp = getattr(modules, cfg.models.hyper.type)(
                    num_layers=cfg.models.hyper.num_layers,
                    hidden_size=cfg.models.hyper.hidden_size,
                    skip_connect_every=cfg.models.hyper.skip_connect_every,
                    num_encoding_fn_xyz=cfg.models.hyper.num_encoding_fn_xyz,
                    include_driving=cfg.models.hyper.include_driving,
                    include_pose=False,
                    ambient_coord_dim=cfg.models.hyper.ambient_coord_dim,
                )
            # elif self.hyper_slice_method == 'axis_aligned_plane':
            # Not implemented

        nerf_mlps = torch.nn.ModuleDict({
            'coarse': getattr(modules, cfg.models.coarse.type)(
                num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
                num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
                num_encoding_fn_ambient=cfg.models.hyper.num_encoding_fn_ambient,
                include_input_xyz=cfg.models.coarse.include_input_xyz,
                include_input_dir=cfg.models.coarse.include_input_dir,
                include_input_ambient=cfg.models.hyper.include_input_ambient,
                use_viewdirs=cfg.models.coarse.use_viewdirs,
                use_ambient=cfg.models.hyper.use_ambient,
                use_pose=cfg.models.coarse.use_pose,
                include_pose=cfg.models.coarse.include_pose,
                use_spatial_embeddings=cfg.models.coarse.use_spatial_embeddings,
                num_layers=cfg.models.coarse.num_layers,
                hidden_size=cfg.models.coarse.hidden_size,
                include_driving=cfg.models.coarse.include_driving,
                ambient_coord_dim=cfg.models.hyper.ambient_coord_dim,
                latent_code_dim=0,
                spatial_embedding_dim=32,
            ),
            'fine': getattr(modules, cfg.models.fine.type)(
                num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
                num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
                num_encoding_fn_ambient=cfg.models.hyper.num_encoding_fn_ambient,
                include_input_xyz=cfg.models.fine.include_input_xyz,
                include_input_dir=cfg.models.fine.include_input_dir,
                include_input_ambient=cfg.models.hyper.include_input_ambient,
                use_viewdirs=cfg.models.fine.use_viewdirs,
                use_ambient=cfg.models.hyper.use_ambient,
                use_pose=cfg.models.coarse.use_pose,
                include_pose=cfg.models.coarse.include_pose,
                use_spatial_embeddings=cfg.models.coarse.use_spatial_embeddings,
                num_layers=cfg.models.coarse.num_layers,
                hidden_size=cfg.models.coarse.hidden_size,
                include_driving=cfg.models.fine.include_driving,
                ambient_coord_dim=cfg.models.hyper.ambient_coord_dim,
                latent_code_dim=0,
                spatial_embedding_dim=32,
            ) if hasattr(cfg.models, "fine") else None
        })

        self.nerf_mlps = nerf_mlps

    def map_spatial_points(self, points, driving=None, pose=None):
        warped_points = points
        points_embed = self.encode_position_fn(points)
        dx = self.warp_field_mlp(points_embed, driving=driving, pose=pose)
        warped_points = warped_points + dx
        return warped_points

    def map_hyper_points(self, points, driving=None, pose=None):
        hyper_points = None
        points_embed = self.encode_position_fn(points)
        if self.hyper_slice_method == 'bendy_sheet':
            hyper_points = self.hyper_sheep_mlp(points_embed, driving, pose)
        # elif self.hyper_slice_method == 'axis_aligned_plane':
        # Not implemented
        # hyper_points =
        return hyper_points

    def map_points(self, points, driving=None, pose=None):
        spatial_points = points
        hyper_points = None
        if self.use_warp:
            spatial_points = self.map_spatial_points(points, driving, pose)
        if self.use_ambient:
            hyper_points = self.map_hyper_points(points, driving, pose)
        if hyper_points is not None:
            mapped_points = torch.cat((spatial_points, hyper_points), dim=-1)
        else:
            mapped_points = spatial_points
        return mapped_points

    def query_template(self, level, points, viewdirs, driving=None, pose=None, latent_code=None, spatial_embedding=None):
        points_embed = self.encode_position_fn(points[..., :3])

        if points.shape[-1] > 3:
            hyper_embed = self.encode_ambient_fn(points[..., 3:])
            points_embed = torch.cat((points_embed, hyper_embed), dim=-1)

        dirs_embed = None
        if self.use_viewdirs:
            dirs_embed = self.encode_direction_fn(viewdirs)

        raw = self.nerf_mlps[level](points_embed, dirs_embed, driving=driving, pose=pose, latent_code=latent_code,
                                    spatial_embedding=spatial_embedding)
        return raw

    def sample_from_3dgrid(self, level, coordinates):
        """
        Expects coordinates in shape (batch_size, num_points_per_batch, 3)
        Expects grid in shape (1, channels, H, W, D)
        (Also works if grid has batch size)
        Returns sampled features of shape (batch_size * num_points_per_batch, feature_channels)
        """
        n_coords = self.num_coarse
        if level == 'fine':
            n_coords += self.num_fine
        coordinates = coordinates.float()

        coordinates = coordinates.reshape((-1, n_coords, coordinates.shape[-1]))
        batch_size, n_coords, n_dims = coordinates.shape
        sampled_features = torch.nn.functional.grid_sample(self.spatial_embeddings.expand(batch_size, -1, -1, -1, -1),
                                                           coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                           mode='bilinear', padding_mode='zeros', align_corners=True)
        N, C, H, W, D = sampled_features.shape
        sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N * H * W * D, C)
        return sampled_features

    def forward(self, level, x, driving=None, pose=None, pose_c=None, latent_code=None, **kwargs):
        # Stage 01
        xyz, viewdirs = x[..., :3], x[..., 3:6]
        driving = driving.repeat(x.shape[0], 1)
        pose = pose_to_euler_trans(pose.unsqueeze(0), pose.device)
        pose = self.encode_pose_fn(pose)
        pose = pose.repeat(x.shape[0], 1)
        mapped_points = self.map_points(xyz, driving, pose)
        spatial_embedding = None
        if self.use_spatial_embeddings and self.spatial_embeddings is not None:
            spatial_embedding = self.sample_from_3dgrid(level, mapped_points[..., :3])
        out = self.query_template(level, mapped_points, viewdirs, driving=driving, latent_code=latent_code,
                                  spatial_embedding=spatial_embedding)
        return out

        # stage 2
        # xyz, viewdirs = x[..., :3], x[..., 3:6]
        #
        # index = torch.linspace(
        #     0,
        #     x.shape[0],
        #     x.shape[0],
        #     dtype=torch.int32,
        #     device=x.device)
        # xyz_i = torch.cat((xyz, index[:, None]), dim=-1)
        # mask = x[..., 6:]
        #
        # inHead = torch.sum(mask[:, 1:10], dim=1)
        # xyz_inHead = xyz_i[inHead == 1, :]
        # pose_inHead = pose_to_euler_trans(pose_c.unsqueeze(0), pose_c.device)
        # pose_inHead = self.encode_pose_fn(pose_inHead)
        # pose_inHead = pose_inHead.repeat((xyz_inHead.shape[0], 1))
        #
        # inTorso = mask[:, 0] + torch.sum(mask[:, 10:12], dim=1)
        # xyz_inTorso = xyz_i[inTorso == 1, :]
        # pose_inTorso = pose_to_euler_trans(pose.unsqueeze(0), pose.device)
        # pose_inTorso = self.encode_pose_fn(pose_inTorso)
        # pose_inTorso = pose_inTorso.repeat((xyz_inTorso.shape[0], 1))
        #
        # xyz_i = torch.cat((xyz_inHead, xyz_inTorso), dim=0)
        # idx = xyz_i[:, -1].argsort()
        #
        # pose = torch.cat((pose_inHead, pose_inTorso), dim=0)
        # pose = pose[idx]
        #
        # driving = driving.repeat((xyz.shape[0], 1))
        #
        # mapped_points = self.map_points(xyz, driving, pose)
        # spatial_embedding = None
        # if self.use_spatial_embeddings and self.spatial_embeddings is not None:
        #     spatial_embedding = self.sample_from_3dgrid(level, mapped_points[...,:3])
        # out = self.query_template(level, mapped_points, viewdirs, driving=driving, pose=pose, latent_code=latent_code,
        #                           spatial_embedding=spatial_embedding)
        # return out

        # index = torch.linspace(
        #     0,
        #     x.shape[0],
        #     x.shape[0],
        #     dtype=torch.int32,
        #     device=x.device)
        # xyz, viewdirs = x[..., :3], x[..., 3:6]
        # xyz = torch.cat((xyz, index[:, None]), dim=-1)
        # inHead = x[..., -1]
        # xyz_inHead = xyz[inHead == 1, :]
        # xyz_not_inHead = xyz[inHead == 0, :]
        # 头部内的坐标经过水平集和变形场1,头部外的坐标经过变形场2
        # mapped_points = self.map_points(xyz_inHead[:,:-1], expr)
        # warp_points = self.map_points_2(xyz_not_inHead[:,:-1], expr, pose)
        # xyz_inHead = torch.cat((mapped_points, xyz_inHead[:, None, -1]), dim=-1)
        # xyz_not_inHead = torch.cat((warp_points, xyz_not_inHead[:, None, -1]), dim=-1)
        # # xyz_not_inHead = torch.cat((torch.cat((warp_points, torch.zeros((warp_points.shape[0],1),device=warp_points.device)), dim=-1), xyz_not_inHead[:, None, -1]), dim=-1)
        # # if self.use_ambient:
        # #     if self.hyper_include_expression:
        # #         hyper_coord_in_head = self.map_hyper_points(xyz_inHead[:, :-1], expr)
        # #         hyper_coord_not_in_head = self.map_hyper_points_2(xyz_not_inHead[:, :-1], expr)
        # #     else:
        # #         hyper_coord_in_head = self.map_hyper_points(xyz_inHead[:, :-1])
        # #         hyper_coord_not_in_head = self.map_hyper_points_2(xyz_not_inHead[:, :-1])
        # #     xyz_inHead_hyper = torch.cat((xyz_inHead[:, :-1], hyper_coord_in_head, xyz_inHead[:, None, -1]), dim=-1)
        # #     xyz_not_inHead_hyper = torch.cat((xyz_not_inHead[:, :-1], hyper_coord_not_in_head, xyz_not_inHead[:, None, -1]), dim=-1)
        # # warp_points = self.map_spatial_points(xyz_not_inHead[:, :-1], expr)
        # # # xyz_not_inHead_warp = warp_points
        # # if self.use_ambient:
        # #     xyz_not_inHead_warp = torch.cat((warp_points, torch.zeros((warp_points.shape[0], 2),device=warp_points.device), xyz_not_inHead[:, None, -1]), dim=-1)
        # xyz = torch.cat((xyz_inHead, xyz_not_inHead), dim=0)
        # xyz = xyz[xyz[:,-1].argsort(),:-1]
        # out = self.query_template(level, xyz, viewdirs, expression=expr, latent_code=latent_code,
        #                           spatial_embedding=spatial_embedding)
        # return out

        # xyz = xyz_inHead[..., :3]
        # mapped_points = self.warp_field_mlp(self.encode_position_fn(xyz), expr)
        # xyz_inHead = torch.cat((mapped_points, xyz_inHead[:, None, -1]), dim=-1)
        #
        # xyz = xyz_not_inHead[..., :3]
        # mapped_points = self.warp_field_mlp_not_in_head(self.encode_position_fn(xyz))
        # xyz_not_inHead = torch.cat((mapped_points, xyz_not_inHead[:, None, -1]), dim=-1)
        #
        # xyz_warped = torch.cat((xyz_inHead, xyz_not_inHead), dim=0)
        # xyz_warped = xyz_warped[torch.argsort(xyz_warped[:, 3]), :3]
        # hyper_points = None
        # if self.use_ambient:
        #     if self.hyper_include_expression:
        #         hyper_points = self.map_hyper_points(xyz_warped, expr)
        #     else:
        #         hyper_points = self.map_hyper_points(xyz_warped)
        # if hyper_points is not None:
        #     mapped_points = torch.cat((xyz_warped, hyper_points), dim=-1)
        # else:
        #     mapped_points = xyz_warped
        # out = self.query_template(level, mapped_points, viewdirs, expression=expr, latent_code=latent_code, spatial_embedding=spatial_embedding)
        # return out


def rot_to_euler(R, device):
    batch_size, _, _ = R.shape
    e = torch.ones((batch_size, 3)).to(device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]
    e[:, 2] = torch.atan2(R00, -R01)
    e[:, 1] = torch.asin(-R02)
    e[:, 0] = torch.atan2(R22, R12)
    return e


def pose_to_euler_trans(poses, device):
    e = rot_to_euler(poses, device)
    t = poses[:, :3, 3]
    return torch.cat((e, t), dim=1)


class AudioFaceModel(NeRFaceModel):
    def __init__(self, cfg):
        super(AudioFaceModel, self).__init__(cfg)
        from .modules import AudioNet
        self.audNet_head = AudioNet(76, 16)
        # self.audNet_tensor = AudioNet(76, 16)

    def forward(self, level, x, audio=None, pose=None, pose_c=None, latent_code=None, **kwargs):
        # Stage 01
        xyz, viewdirs = x[..., :3], x[..., 3:6]
        driving = self.audNet_head(audio.unsqueeze(0))
        driving = driving.repeat(x.shape[0], 1)
        pose = pose_to_euler_trans(pose.unsqueeze(0), pose.device)
        pose = self.encode_pose_fn(pose)
        pose = pose.repeat(x.shape[0], 1)
        mapped_points = self.map_points(xyz, driving, pose)
        spatial_embedding = None
        if self.use_spatial_embeddings and self.spatial_embeddings is not None:
            spatial_embedding = self.sample_from_3dgrid(level, mapped_points[..., :3])
        out = self.query_template(level, mapped_points, viewdirs, driving=driving, pose=pose, latent_code=latent_code,
                                  spatial_embedding=spatial_embedding)
        return out

        # Stage 02
        # xyz, viewdirs = x[..., :3], x[..., 3:6]
        #
        # driving_inHead = self.audNet_head(audio.unsqueeze(0))
        # driving_inTorso = self.audNet_tensor(audio.unsqueeze(0))
        #
        # index = torch.linspace(
        #     0,
        #     x.shape[0],
        #     x.shape[0],
        #     dtype=torch.int32,
        #     device=x.device)
        # xyz_i = torch.cat((xyz, index[:, None]), dim=-1)
        # mask = x[..., 6:]
        #
        # inHead = torch.sum(mask[:, 1:10], dim=1)
        # xyz_inHead = xyz_i[inHead == 1, :]
        # driving_inHead = driving_inHead.repeat((xyz_inHead.shape[0], 1))
        # pose_inHead = pose_to_euler_trans(pose_c.unsqueeze(0), pose_c.device)
        # pose_inHead = self.encode_pose_fn(pose_inHead)
        # pose_inHead = pose_inHead.repeat((xyz_inHead.shape[0], 1))
        #
        # inTorso = mask[:, 0] + torch.sum(mask[:, 10:12], dim=1)
        # xyz_inTorso = xyz_i[inTorso == 1, :]
        # driving_inTorso = driving_inTorso.repeat((xyz_inTorso.shape[0], 1))
        # pose_inTorso = pose_to_euler_trans(pose.unsqueeze(0), pose.device)
        # pose_inTorso = self.encode_pose_fn(pose_inTorso)
        # pose_inTorso = pose_inTorso.repeat((xyz_inTorso.shape[0], 1))
        #
        # xyz_i = torch.cat((xyz_inHead, xyz_inTorso), dim=0)
        # idx = xyz_i[:, -1].argsort()
        #
        # driving = torch.cat((driving_inHead, driving_inTorso), dim=0)
        # driving = driving[idx]
        #
        # pose = torch.cat((pose_inHead, pose_inTorso), dim=0)
        # pose = pose[idx]
        #
        # mapped_points = self.map_points(xyz, driving, pose)
        # spatial_embedding = None
        # if self.use_spatial_embeddings and self.spatial_embeddings is not None:
        #     spatial_embedding = self.sample_from_3dgrid(level, mapped_points[..., :3])
        # out = self.query_template(level, mapped_points, viewdirs, driving=driving, pose=pose, latent_code=latent_code,
        #                           spatial_embedding=spatial_embedding)
        # return out


class AudioMaskGenerator(NeRFaceModel):
    def __init__(self, cfg):
        super(AudioMaskGenerator, self).__init__(cfg)
        self.encode_pose_fn = get_embedding_function(
            num_encoding_functions=3,
            include_input=False,
            log_sampling=True,
        )
        from .modules import AudioNet
        self.audNet_head = AudioNet(76, 16)
        self.audNet_tensor = AudioNet(76, 16)

    def forward(self, level, x, audio=None, pose=None, latent_code=None, **kwargs):
        xyz, viewdirs = x[..., :3], x[..., 3:6]
        spatial_embedding = None
        if self.use_spatial_embeddings and self.spatial_embeddings is not None:
            spatial_embedding = self.sample_from_3dgrid(level, xyz)
            # spatial_embedding = spatial_embedding.reshape((-1, spatial_embedding.shape[-1]))
        driving = self.audNet_head(audio.unsqueeze(0))
        driving = driving.repeat(x.shape[0], 1)
        pose = pose_to_euler_trans(pose.unsqueeze(0), pose.device)
        pose = self.encode_pose_fn(pose)
        pose = pose.repeat(x.shape[0], 1)

        mapped_points = self.map_points(xyz, driving, pose)
        out = self.query_template(level, mapped_points, viewdirs, driving=driving, latent_code=latent_code,
                                  spatial_embedding=spatial_embedding)
        return out


class ConditionalBlendshapePaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            num_encoding_fn_ambient=4,
            include_input_xyz=True,
            include_input_dir=True,
            include_input_ambient=True,
            use_viewdirs=True,
            use_ambient=True,
            include_expression=True,
            ambient_coord_dim=2,
            latent_code_dim=32,
            encode_ambient_fn=None,
    ):
        super(ConditionalBlendshapePaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0
        include_input_ambient = ambient_coord_dim if include_input_ambient else 0

        self.encode_ambient_fn = encode_ambient_fn

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        self.dim_ambient = include_input_ambient + 2 * include_input_ambient * num_encoding_fn_ambient
        self.dim_latent_code = latent_code_dim

        self.layers_ambient = torch.nn.ModuleList()
        self.use_ambient = use_ambient
        self.layers_ambient.append(torch.nn.Linear(self.dim_xyz, 64))
        for i in range(1, 6):
            if i == 4:
                self.layers_ambient.append(torch.nn.Linear(self.dim_xyz + 64, 64))
            else:
                self.layers_ambient.append(torch.nn.Linear(64, 64))
        self.fc_ambient = torch.nn.Linear(64, include_input_ambient)

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(
            torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_ambient + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_ambient + self.dim_latent_code + 256,
                                    256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x = xyz  # self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.use_ambient:
            w = x
            for i in range(6):
                if i == 4:
                    w = self.layers_ambient[i](torch.cat((w, x), -1))
                else:
                    w = self.layers_ambient[i](w)
                w = self.relu(w)
            ambient = self.fc_ambient(w)
        if self.encode_ambient_fn is not None:
            ambient = self.encode_ambient_fn(ambient)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            if self.use_ambient:
                initial = torch.cat((xyz, expr_encoding, ambient, latent_code), dim=1)
            else:
                initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class ConditionalBlendshapePaperSmallerNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """  # Made smaller...

    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True,
            latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperSmallerNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 5):
            if i == 3:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir + self.dim_expression, 128))
        for i in range(2):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x = xyz  # self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(5):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs, expr_encoding), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)


class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, x):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True
    ):
        super(ConditionalNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 1 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        self.dim_expression = 0  # 15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        # self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 4

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = expr.repeat(xyz.shape[0], 1) * (1 / 3)
            # if self.layers_expr is not None:
            #    for l in self.layers_expr:
            #        expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = self.sigmoid(expr_encoding)
            # expr_encoding = self.layers_expr[1](expr_encoding)
            # expr_encoding = self.sigmoid(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalBlendshapeLearnableCodeNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True,
            latent_code_dim=32
    ):
        super(ConditionalBlendshapeLearnableCodeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        # self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        self.dim_latent_code = latent_code_dim
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        self.layers_expr = None
        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 2

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):  # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression + self.dim_latent_code,
                                    hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            # if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
                expr_encoding = self.layers_expr[0](expr_encoding)
                expr_encoding = torch.nn.functional.tanh(expr_encoding)
                # expr_encoding = self.layers_expr[1](expr_encoding)
                # expr_encoding = self.relu(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding, latent_code), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz, expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalCompressedBlendshapeLearnableCodeNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True,
            latent_code_dim=32
    ):
        super(ConditionalCompressedBlendshapeLearnableCodeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        # self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        self.dim_latent_code = latent_code_dim
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        self.dim_expression = 10
        self.layer_expr = torch.nn.Linear(76, 10)
        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 2

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):  # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression + self.dim_latent_code,
                                    hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (self.layer_expr(expr)).repeat(xyz.shape[0], 1)
            # if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = torch.nn.functional.tanh(expr_encoding)
            #    #expr_encoding = self.layers_expr[1](expr_encoding)
            #    #expr_encoding = self.relu(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding, latent_code), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz, expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalCompressedBlendshapeNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True
    ):
        super(ConditionalCompressedBlendshapeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 20 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        # self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        self.dim_expression = 20
        # self.layer_expr = torch.nn.Linear(76,10)

        self.layers_expr = torch.nn.ModuleList()

        self.layers_expr.append(torch.nn.Linear(76, 38))
        self.layers_expr.append(torch.nn.Linear(38, 20))
        self.layers_expr.append(torch.nn.Linear(20, 20))

        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        # self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 2

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):  # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        # latent_code = latent_code.repeat(xyz.shape[0],1)
        if self.dim_expression > 0:
            expr = expr.repeat(xyz.shape[0], 1)

            for expr_layer in self.layers_expr:
                expr = expr_layer(expr)
                expr = self.relu(expr)
            # if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = torch.nn.functional.tanh(expr_encoding)
            #    #expr_encoding = self.layers_expr[1](expr_encoding)
            #    #expr_encoding = self.relu(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            # expr = expr.repeat(xyz.shape[0], 1)
            x = torch.cat((xyz, expr), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz, expr), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalBlendshapeNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True
    ):
        super(ConditionalBlendshapeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        # self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        self.layers_expr = None
        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 2

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):  # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            # if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
                expr_encoding = self.layers_expr[0](expr_encoding)
                expr_encoding = torch.nn.functional.tanh(expr_encoding)
                # expr_encoding = self.layers_expr[1](expr_encoding)
                # expr_encoding = self.relu(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz, expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ConditionalBlendshapeNeRFModel_v2(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True
    ):
        super(ConditionalBlendshapeNeRFModel_v2, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 15 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        # self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        self.layers_expr = torch.nn.ModuleList()
        # self.layers_expr = None
        self.layers_expr.append(torch.nn.Linear(self.dim_expression, self.dim_expression * 2))
        self.layers_expr.append(torch.nn.Linear(self.dim_expression * 2, self.dim_expression * 4))
        self.dim_expression *= 4

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            # expr_encoding = (expr * 1/3).repeat(xyz.shape[0],1)
            expr_encoding = (expr * 1 / 3)  # .repeat(xyz.shape[0],1)
            # if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
                expr_encoding = self.layers_expr[0](expr_encoding)
                expr_encoding = torch.nn.functional.relu(expr_encoding)
                expr_encoding = self.layers_expr[1](expr_encoding)
                expr_encoding = self.relu(expr_encoding)
                expr_encoding = expr_encoding.repeat(xyz.shape[0], 1)
            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.n_down = 5
        # Bx3x256x256 -> Bx128x1x1
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 64

            torch.nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 16

            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 4

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 1

            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        return x


class ConditionalAutoEncoderNeRFModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=128,
            skip_connect_every=4,
            num_encoding_fn_xyz=6,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_expression=True
    ):
        super(ConditionalAutoEncoderNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 128 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression  # + 2 * 3 * num_encoding_fn_expr
        self.dim_expression = 0  # 15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        # Encoding for expressions:
        # self.layers_expr = torch.nn.ModuleList()
        # self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        # self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        # self.dim_expression *= 4

        # self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = expr.repeat(xyz.shape[0], 1)
            # if self.layers_expr is not None:
            #    for l in self.layers_expr:
            #        expr_encoding = self.relu(l(expr_encoding))
            # if self.layers_expr is not None:
            #    expr_encoding = self.layers_expr[0](expr_encoding)
            #    expr_encoding = self.sigmoid(expr_encoding)
            # expr_encoding = self.layers_expr[1](expr_encoding)
            # expr_encoding = self.sigmoid(expr_encoding)

            # x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz)):
            if (
                    i % self.skip_connect_every == 0
                    and i > 0
                    and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class DiscriminatorModel(torch.nn.Module):
    def __init__(self, dim_latent=32, dim_expressions=76):
        super(DiscriminatorModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, dim_latent * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent * 2, dim_latent * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent * 2, dim_expressions),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
