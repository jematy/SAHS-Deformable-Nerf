import torch.nn
import numpy as np


# Audio feature extractor
class AudioAttNet(torch.nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = torch.nn.Sequential(  # b x subspace_dim x seq_len
            torch.nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                            stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.seq_len,
                            out_features=self.seq_len, bias=True),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = x[..., :self.dim_aud].permute(1, 0).unsqueeze(
            0)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1)
        # print(y.view(-1).data)
        return torch.sum(y * x, dim=0)


# Model


# Audio feature extractor
class AudioNet(torch.nn.Module):
    def __init__(self, dim_aud=76, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = torch.nn.Sequential(  # n x 29 x 16
            torch.nn.Conv1d(29, 32, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 32 x 8
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(32, 32, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 32 x 4
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(32, 64, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 64 x 2
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=2,
                            padding=1, bias=True),  # n x 64 x 1
            torch.nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(0.02, True),
            torch.nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, 8 - half_w:8 + half_w, :].permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).squeeze()
        return x


class MaskGeneratorMLP(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
            self,
            num_layers=6,
            hidden_size=256,
            skip_connect_every=4,
            num_encoding_fn_xyz=10,
            num_encoding_fn_dir=4,
            include_input_xyz=True,
            include_input_dir=True,
            use_viewdirs=True,
            include_driving=True,
            latent_code_dim=32,
    ):
        super(MaskGeneratorMLP, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_driving = 76 if include_driving else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.use_viewdirs = use_viewdirs

        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_driving = include_driving  # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim

        self.layers_xyz = torch.nn.ModuleList()
        input_dim = self.dim_xyz + self.dim_latent_code
        if self.dim_driving > 0:
            input_dim += self.dim_driving

        self.layers_xyz.append(torch.nn.Linear(input_dim, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(torch.nn.Linear(input_dim + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 256))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(256, 256))
        self.fc_rgb = torch.nn.Linear(256, 3)

        self.layers_seg = torch.nn.ModuleList()
        for i in range(4):
            self.layers_seg.append(torch.nn.Linear(256, 256))
        self.fc_seg = torch.nn.Linear(256, 1)

        self.relu = torch.nn.LeakyReLU(0.01)

    def forward(self, xyz, dirs, driving=None, latent_code=None):
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        initial = torch.cat((xyz, latent_code), dim=1)
        x = initial
        if self.dim_driving > 0:
            # drive_encoding = (driving * 1 / 3).repeat(xyz.shape[0], 1)
            # drive_encoding = driving.repeat(xyz.shape[0], 1)
            initial = torch.cat((x, driving), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((x, initial), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)

        for i in range(4):
            x = self.layers_seg[i](feat)
            x = self.relu(x)
        seg = self.fc_seg(x)

        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, seg, alpha), dim=-1)


class NeRFMLP(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
            self,
            num_layers=6,
            hidden_size=256,
            skip_connect_every=3,
            num_encoding_fn_xyz=10,
            num_encoding_fn_dir=4,
            num_encoding_fn_ambient=4,
            include_input_xyz=False,
            include_input_dir=False,
            use_viewdirs=False,
            use_ambient=False,
            use_pose=False,
            use_spatial_embeddings=False,
            include_driving=False,
            include_input_ambient=False,
            include_pose=False,
            latent_code_dim=32,
            spatial_embedding_dim=32,
            ambient_coord_dim=1
    ):
        super(NeRFMLP, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.skip_connect_every = skip_connect_every

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_input_ambient = ambient_coord_dim if include_input_ambient else 0
        include_driving = 76 if include_driving else 0
        include_pose = 6 if include_pose else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.use_ambient = use_ambient
        self.use_viewdirs = use_viewdirs
        self.use_pose = use_pose
        self.use_spatial_embeddings = use_spatial_embeddings

        if self.use_ambient:
            self.dim_ambient = include_input_ambient + 2 * ambient_coord_dim * num_encoding_fn_ambient
            self.dim_xyz += self.dim_ambient

        if self.use_pose:
            self.dim_pose = include_pose + 2 * 6 * 3
            self.dim_xyz += self.dim_pose

        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_driving = include_driving  # + 2 * 3 * num_encoding_fn_expr
        self.dim_latent_code = latent_code_dim
        self.dim_spatial_embedding = spatial_embedding_dim if self.use_spatial_embeddings else 0

        self.layers_xyz = torch.nn.ModuleList()
        input_dim = self.dim_xyz + self.dim_latent_code
        if self.dim_driving > 0:
            input_dim += self.dim_driving

        self.layers_xyz.append(torch.nn.Linear(input_dim, self.hidden_size))
        for i in range(1, self.num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(input_dim + self.hidden_size, self.hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
        self.fc_feat = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_alpha = torch.nn.Linear(self.hidden_size, 1)

        rgb_hidden_size = self.hidden_size // 2
        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(self.hidden_size + self.dim_dir + self.dim_spatial_embedding, rgb_hidden_size))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(rgb_hidden_size, rgb_hidden_size))
        self.fc_rgb = torch.nn.Linear(rgb_hidden_size, 3)

        seg_hidden_size = self.hidden_size // 2
        self.layers_seg = torch.nn.ModuleList()
        self.layers_seg.append(torch.nn.Linear(self.hidden_size, seg_hidden_size))
        for i in range(3):
            self.layers_seg.append(torch.nn.Linear(seg_hidden_size, seg_hidden_size))
        self.fc_seg = torch.nn.Linear(seg_hidden_size, 12)

        self.relu = torch.nn.LeakyReLU(0.01)

    def forward(self, xyz, dirs, driving=None, pose=None, latent_code=None, spatial_embedding=None):
        initial = xyz
        if latent_code is not None:
            latent_code = latent_code.repeat(xyz.shape[0], 1)
            initial = torch.cat((initial, latent_code), dim=1)
        x = initial
        if self.dim_driving > 0:
            # drive_encoding = (driving * 1 / 3).repeat(xyz.shape[0], 1)
            # drive_encoding = driving.repeat(xyz.shape[0], 1)
            initial = torch.cat((x, driving), dim=1)
            x = initial
        if self.use_pose:
            initial = torch.cat((x, pose), dim=1)
            x = initial
        for i in range(self.num_layers):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((x, initial), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            if self.use_spatial_embeddings and spatial_embedding is not None:
                x = self.layers_dir[0](torch.cat((feat, dirs, spatial_embedding), -1))
            else:
                x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, 4):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)

        x = self.layers_seg[0](feat)
        x = self.relu(x)
        for i in range(1, 4):
            x = self.layers_seg[i](x)
            x = self.relu(x)
        seg = self.fc_seg(x)
        return torch.cat((rgb, seg, alpha), dim=-1)


class WarpEmbeddingMLP(torch.nn.Module):
    def __init__(
            self,
            num_layers=4,
            hidden_size=64,
            input_s=36,
            output_s=36
    ):
        super(WarpEmbeddingMLP, self).__init__()

        self.num_layers = num_layers

        self.layers_ambient = torch.nn.ModuleList()
        self.layers_ambient.append(torch.nn.Linear(input_s, hidden_size))
        for i in range(1, num_layers-1):
            self.layers_ambient.append(torch.nn.Linear(hidden_size, hidden_size))
        self.layers_ambient.append(torch.nn.Linear(hidden_size, output_s))
        self.relu = torch.relu

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers_ambient[i](x)
            x = self.relu(x)
        return x

class WarpFieldMLP(torch.nn.Module):
    """Network that predicts warps as an SE(3) field."""

    def __init__(
            self,
            num_layers=6,
            hidden_size=64,
            skip_connect_every=4,
            num_encoding_fn_xyz=10,
            include_input_xyz=True,
            include_driving=True,
            include_pose=True,
    ):
        super(WarpFieldMLP, self).__init__()

        self.num_layers = num_layers
        self.skip_connect_every = skip_connect_every

        include_input_xyz = 3 if include_input_xyz else 0
        include_driving = 76 if include_driving else 0
        include_pose = 6 if include_pose else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_driving = include_driving
        self.dim_pose = include_pose + 2 * 6 * 3

        # if include_expression:
        #     self.pre = torch.nn.Linear(self.dim_expression, 16)
        #     self.dim_expression = 16

        self.layers_xyz = torch.nn.ModuleList()
        input_dim = self.dim_xyz
        if self.dim_driving > 0:
            input_dim += self.dim_driving
        if self.dim_pose > 0:
            input_dim += self.dim_pose

        self.layers_xyz.append(torch.nn.Linear(input_dim, hidden_size))
        for i in range(1, num_layers):
            if i == skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(input_dim + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.fc_final = torch.nn.Linear(hidden_size, 3)
        self.relu = torch.nn.functional.relu
        self.activation = torch.tanh

    def forward(self, x, driving=None, pose=None):
        initial = x
        if self.dim_driving > 0:
            # drive_encoding = (driving * 1 / 3).repeat(x.shape[0], 1)
            # drive_encoding = driving.repeat(x.shape[0], 1)
            initial = torch.cat((x, driving), dim=1)
            x = initial
        if self.dim_pose > 0:
            # pose_encoding = pose.repeat(x.shape[0], 1)
            initial = torch.cat((x, pose), dim=1)
            x = initial
        for i in range(self.num_layers):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((x, initial), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        x = self.activation(self.fc_final(x))
        # dx, dsigma = x[..., :3], x[..., 3:4]
        return x  # dx, dsigma
        # w = self.relu(self.fc_rotation(x))
        # v = self.relu(self.fc_translation(x))
        # theta = torch.linalg.norm(w, dim=-1)
        # w = w / theta.unsqueeze(-1)
        # v = v / theta.unsqueeze(-1)
        # screw_axis = torch.cat([w, v], -1)  # (128,6)
        # transform = rigid.exp_se3(screw_axis, theta)  # (128,4,4)
        # return transform


class HyperSheetMLP(torch.nn.Module):
    """An MLP that defines a bendy slicing surface through hyper space."""

    def __init__(
            self,
            num_layers=6,
            hidden_size=64,
            skip_connect_every=4,
            num_encoding_fn_xyz=10,
            include_input_xyz=True,
            include_driving=True,
            include_pose=True,
            ambient_coord_dim=1,
    ):
        super(HyperSheetMLP, self).__init__()

        self.num_layers = num_layers
        self.skip_connect_every = skip_connect_every

        include_input_xyz = 3 if include_input_xyz else 0
        include_driving = 76 if include_driving else 0
        include_pose = 6 if include_pose else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_driving = include_driving
        self.dim_pose = include_pose + 2 * 6 * 3

        input_dim = self.dim_xyz
        if self.dim_driving > 0:
            input_dim += self.dim_driving
        if self.dim_pose > 0:
            input_dim += self.dim_pose

        self.layers_ambient = torch.nn.ModuleList()
        self.layers_ambient.append(torch.nn.Linear(input_dim, hidden_size))
        for i in range(1, num_layers):
            if i == skip_connect_every:
                self.layers_ambient.append(torch.nn.Linear(input_dim + hidden_size, hidden_size))
            else:
                self.layers_ambient.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_ambient = torch.nn.Linear(hidden_size, ambient_coord_dim)
        self.relu = torch.nn.functional.relu

    def forward(self, w, driving=None, pose=None):
        initial = w
        if self.dim_driving > 0:
            # drive_encoding = (driving * 1 / 3).repeat(w.shape[0], 1)
            # drive_encoding = driving.repeat(w.shape[0], 1)
            initial = torch.cat((w, driving), dim=1)
            w = initial
        if self.dim_pose > 0:
            # pose_encoding = pose.repeat(x.shape[0], 1)
            initial = torch.cat((w, pose), dim=1)
            w = initial
        for i in range(self.num_layers):
            if i == self.skip_connect_every:
                w = self.layers_ambient[i](torch.cat((w, initial), -1))
            else:
                w = self.layers_ambient[i](w)
            w = self.relu(w)
        ambient = self.fc_ambient(w)
        return ambient
