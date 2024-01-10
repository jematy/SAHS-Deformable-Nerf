import torch

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .nerf_helpers import dump_rays
from .volume_rendering_utils import volume_render_radiance_field


def run_network(level, network_fn, pts, ray_batch, chunksize, use_viewdirs, driving=None, pose=None, pose_c=None, latent_code=None, spatial_embeddings=None):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    pts_dirs = pts_flat

    # embedded = embed_fn(pts_flat)
    if use_viewdirs:
        viewdirs = ray_batch[..., None, 3:6]
        # viewdirs = ray_batch[..., None, 5:8]
        input_dirs = viewdirs.expand(pts[..., :3].shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        # embedded_dirs = embeddirs_fn(input_dirs_flat)
        # embedded = torch.cat((embedded, embedded_dirs), dim=-1)
        pts_dirs = torch.cat((pts_flat, input_dirs_flat), dim=-1)

    if ray_batch.shape[-1] > 8:
        inHead = ray_batch[..., None, 8:].expand([pts.shape[0], pts.shape[1], ray_batch.shape[-1]-8])
        inTorso_flat = inHead.reshape((-1, ray_batch.shape[-1]-8))
        pts_dirs = torch.cat((pts_dirs, inTorso_flat), dim=-1)

    # if spatial_embeddings is not None:
    #     spatial_embedding = sample_from_3dgrid(pts * 2 / 0.24, spatial_embeddings)
    #     spatial_embedding = spatial_embedding.reshape((-1, spatial_embedding.shape[-1]))
    #     spatial_embedding = get_minibatches(spatial_embedding, chunksize=chunksize)

    batches = get_minibatches(pts_dirs, chunksize=chunksize)
    preds = [network_fn(level, batch, driving, pose, pose_c, latent_code=latent_code) for batch in batches]
    # if driving is None:
    #     preds = [network_fn(level, batch) for batch in batches]
    # elif spatial_embeddings is not None:
    #     preds = [network_fn(level, b, driving, pose, latent_code=latent_code, spatial_embedding=se) for b, se in zip(batches, spatial_embedding)]
    # elif latent_code is not None:
    #     preds = [network_fn(level, batch, driving, pose, latent_code=latent_code) for
    #              batch in batches]
    # else:
    #     preds = [network_fn(level, batch, driving) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    del pts_dirs, input_dirs_flat
    return radiance_field


# def sample_from_3dgrid(coordinates, grid):
#     """
#     Expects coordinates in shape (batch_size, num_points_per_batch, 3)
#     Expects grid in shape (1, channels, H, W, D)
#     (Also works if grid has batch size)
#     Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
#     """
#     coordinates = coordinates.float()
#     grid = grid.float()
#
#     batch_size, n_coords, n_dims = coordinates.shape
#     sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
#                                                        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
#                                                        mode='bilinear', padding_mode='zeros', align_corners=True)
#     N, C, H, W, D = sampled_features.shape
#     sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
#     return sampled_features


def predict_and_render_radiance(
        ray_batch,
        model,
        options,
        mode="train",
        driving=None,
        pose=None,
        pose_c=None,
        background_prior=None,
        latent_code=None,
        spatial_embeddings=None,
        ray_dirs_fake=None
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone()  # TODO remove clone ablation rays
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    # Uncomment to dump a ply file visualizing camera rays and sampling points
    # dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    # ray_batch[..., 3:6] = ray_dirs_fake[0][..., 3:6]  # TODO remove this this is for ablation of ray dir
    # 2048, 64, 4
    radiance_field = run_network(
        'coarse',
        model,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        options.nerf.use_viewdirs,
        driving,
        pose,
        pose_c,
        latent_code,
        spatial_embeddings=spatial_embeddings,
    )
    # make last RGB values of each ray, the background
    if background_prior is not None:
        radiance_field[:, -1, :-1] = background_prior

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        background_prior=background_prior
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            'fine',
            model,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            options.nerf.use_viewdirs,
            driving,
            pose,
            pose_c,
            latent_code,
            spatial_embeddings
        )
        # make last RGB values of each ray, the background
        if background_prior is not None:
            radiance_field[:, -1, :-1] = background_prior

        # Uncomment to dump a ply file visualizing camera rays and sampling points
        # dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), radiance_field)

        # dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), torch.softmax(radiance_field[:,:,-1],1).detach().cpu().numpy())

        # rgb_fine, disp_fine, acc_fine, _, depth_fine = volume_render_radiance_field(
        rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field(  # added use of weights
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior
        )

    # return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine #added depth fine
    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,
                                                                               -1], depth_fine  # changed last return val to fine_weights


def run_one_iter_of_nerf(
        height,
        width,
        focal_length,
        model,
        ray_origins,
        ray_directions,
        options,
        mode="train",
        driving=None,
        pose=None,
        pose_c=None,
        background_prior=None,
        latent_code=None,
        ray_directions_ablation=None,
        spatial_embeddings=None,
        inHead=None
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if hasattr(options.models, "fine"):
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]]
        restore_shapes += [ray_directions.shape[:-1]]  # to return fine depth map
    if options.dataset.no_ndc is False:
        # print("calling ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        # print("calling ndc")
        # "caling normal rays (not NDC)"
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        ray_directions_ablation = ray_directions
        rd_ablations = ray_directions_ablation.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    if inHead is not None:
        inHead = inHead.view((-1, inHead.shape[-1]))
        rays = torch.cat((ro, rd, near, far, inHead), dim=-1)
    else:
        rays = torch.cat((ro, rd, near, far), dim=-1)
    # rays = torch.cat((ro, rd, near, far), dim=-1)
    rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    # if options.nerf.use_viewdirs: # TODO uncomment
    #     rays = torch.cat((rays, viewdirs), dim=-1)
    #
    viewdirs = None  # TODO remove this paragraph
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions_ablation
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

    batches_ablation = get_minibatches(rays_ablation, chunksize=getattr(options.nerf, mode).chunksize)
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    assert (batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if \
        background_prior is not None else background_prior
    # print("predicting")
    pred = [
        predict_and_render_radiance(
            batch,
            model,
            options,
            mode,
            driving=driving,
            pose=pose,
            pose_c=pose_c,
            background_prior=background_prior[i] if background_prior is not None else background_prior,
            latent_code=latent_code,
            spatial_embeddings=spatial_embeddings,
            ray_dirs_fake=batches_ablation
        )
        for i, batch in enumerate(batches)
    ]
    # print("predicted")

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation" and options.models.mask.use_mask and synthesized_images[0].shape[-1] == 15:
        restore_shapes[0] = (restore_shapes[0][0], restore_shapes[0][1], 15)
        restore_shapes[3] = (restore_shapes[3][0], restore_shapes[3][1], 15)
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if hasattr(options.models, "fine"):
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)


def run_one_iter_of_conditional_nerf(
        height,
        width,
        focal_length,
        model_coarse,
        model_fine,
        ray_origins,
        ray_directions,
        expression,
        options,
        mode="train",
        encode_position_fn=None,
        encode_direction_fn=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += ray_directions.shape[:-1]  # for fine depth map

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        for batch in batches
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=5)
