# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import math

import sys

sys.path.insert(0, "../")
from utils import train_utils

# produce rays info for a given camera
def get_ray_bundle(tform_cam2world, resolution=128, sampling="square"):

    # camera parameters
    width = 128
    height = 128
    focal_length = 221.7025

    # define direction of rays from camera parameters
    ii, jj = meshgrid_xy(
        torch.arange(width).cuda().float(), torch.arange(height).cuda().float()
    )
    directions = torch.stack(
        [
            2 * (ii - width * 0.5) / focal_length,
            -2 * (jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )

    # if need to consider lower resolution image
    if resolution != 128:
        # sample lower resolution image, assuming rsolution is a power of 2
        lower = directions.shape[0] // resolution
        directions = directions.view(resolution, lower, resolution, lower, 3)
        directions = directions[:, lower // 2, :, lower // 2]

    ray_directions = torch.sum(
        directions[..., None, :] * tform_cam2world[:3, :3], dim=-1
    )

    # normalize lengths of rays
    lengths = ray_directions ** 2
    lengths = torch.sqrt(lengths.sum(-1))
    ray_directions = ray_directions / lengths.unsqueeze(-1)

    ray_origins = tform_cam2world[:3, -1]

    return ray_origins, ray_directions.view(-1, 3)


def meshgrid_xy(tensor1, tensor2):
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


# get random samples from set of rays
def random_sample_rays(
    ray_origins, ray_directions, num_rays=1000, seed=None, first=[128, 128]
):
    select_inds = np.random.RandomState(seed=seed).choice(
        ray_directions.shape[0], size=(num_rays), replace=False
    )
    select_inds[0] = first[0] * 128 + first[1]

    ray_directions = ray_directions[select_inds]
    return ray_origins, ray_directions


# get samples from set of rays in a square pattern
def square_sample_rays(ray_origins, ray_directions, num_rays=1000):

    resolution = math.sqrt(num_rays)
    coords = torch.stack(
        meshgrid_xy(torch.arange(resolution).cuda(), torch.arange(resolution).cuda()),
        dim=-1,
    )
    coords = coords.reshape((-1, 2))
    coords = coords * 256 // resolution
    coords = coords.long()

    ray_origins = ray_origins[coords[:, 0], coords[:, 1], :]
    ray_directions = ray_directions[coords[:, 0], coords[:, 1], :]
    return ray_origins, ray_directions


# convert ray parameters into ponts sampled along ray
def rays_to_points(origins, directions, num_samples=128):

    # near and far of ray from object bounding box
    diag = 0.72
    near = 0.8 - diag
    far = 0.8 + diag

    near = near * torch.ones_like(directions[..., :1]).cuda()
    far = far * torch.ones_like(directions[..., :1]).cuda()

    t_vals = torch.linspace(0.0, 1.0, num_samples).cuda()
    z_vals = near + t_vals * (far - near)
    z_vals = z_vals.expand([directions.shape[0], num_samples])

    pts = origins.view(1, 1, 3) + directions[..., None, :] * z_vals[..., :, None]
    pts = pts.view(-1, 3)

    # |0.4| is the bounding box of our predictions
    pts_mask = torch.abs(pts) < 0.4
    pts_mask = (pts_mask[:, 0] * pts_mask[:, 1] * pts_mask[:, 2]).int()

    dists = torch.sqrt(((origins - pts) ** 2).sum(-1))

    return pts, pts_mask, dists


# get batch of ray samples
def get_rays(self, locations, resolution=128):
    all_ray_points = []
    all_ray_masks = []
    all_dists = []

    for start_location in locations:
        # get orientation of camera from position
        start_orientation = train_utils.rot_from_positions(start_location)
        world_mat = torch.zeros((4, 4)).cuda()
        world_mat[:3, :3] += start_orientation[:3, :3].transpose(1, 0)
        world_mat[:3, -1] += start_location
        world_mat[-1, -1] += 1

        # get ray paramaters from camera parameters
        ray_origins, ray_directions = get_ray_bundle(world_mat, resolution=resolution)
        # get ray samples from parameters
        ray_points, ray_mask, dists = rays_to_points(
            ray_origins, ray_directions, num_samples=128
        )

        all_ray_points.append(ray_points.view(resolution, resolution, 128, 3))
        all_ray_masks.append(ray_mask.view(resolution, resolution, 128))
        all_dists.append(dists)

    ray_points = torch.stack(all_ray_points)
    ray_masks = torch.stack(all_ray_masks).bool()
    ray_dists = torch.stack(all_dists)
    return ray_points, ray_masks, ray_dists


# subsample batch of rays
def subsample_rays(self, ray_points, ray_masks, imgs, num_points, seed=None):

    bs = imgs.shape[0]
    imgs = imgs.view(bs, 3, -1).permute(0, 2, 1).contiguous()
    ray_points = ray_points.view(bs, -1, 128, 3)
    ray_masks = ray_masks.view(bs, -1, 128)

    i, p, m = [], [], []
    for j in range(bs):
        if seed is not None:
            torch.manual_seed(seed + j)
        idx = torch.randperm(imgs.shape[1])[:num_points]
        i.append(imgs[j, idx])
        p.append(ray_points[j, idx])
        m.append(ray_masks[j, idx])

    i, p, m = torch.stack(i), torch.stack(p), torch.stack(m)
    return i, p.view(bs, -1, 3), m.view(bs, -1)
