# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

# get rotation matrix from camera position
def rot_from_positions(pos):
    camera_direction = torch.nn.functional.normalize(pos, dim=0)

    camera_right = torch.cross(torch.FloatTensor(np.array([0.0, 0.0, 1.0])).cuda(), camera_direction)
    camera_right = torch.nn.functional.normalize(camera_right, dim=0)
    camera_up = torch.cross(camera_direction, camera_right)
    camera_up = torch.nn.functional.normalize(camera_up, dim=0)

    rotation_transform = torch.zeros((4, 4)).cuda()
    rotation_transform[0, :3] += camera_right
    rotation_transform[1, :3] += camera_up
    rotation_transform[2, :3] += camera_direction
    rotation_transform[-1, -1] = 1

    translation_transform = torch.FloatTensor(np.eye(4)).cuda()
    translation_transform[:3, -1] += -pos
    rot = torch.mm(rotation_transform, translation_transform)
    return rot


# get a camera positions which is furthest away from current positions
def get_location_even(self, positions, seed):

    # positions already samples
    cur_positions = torch.FloatTensor(np.array(positions)).cuda().unsqueeze(1)

    # random samples to select over
    possible_positions = torch.FloatTensor(
        self.renderer.random_position(0.8, num=1000, seed=seed)).cuda()
    possible_positions = possible_positions.unsqueeze(0)

    # find furthest from current samples
    diff = ((cur_positions - possible_positions) ** 2).sum(-1)
    diff = torch.min(diff, dim=0)[0]
    furthest_position = torch.argmax(diff)
    new_position = possible_positions[0][furthest_position]

    return new_position.data.cpu().numpy()

# get a random camera position
def get_locations_random(self, positions, seed, dist=0.7):
    possible_positions = torch.FloatTensor(
        self.renderer.random_position(0.8, num=1000, seed=seed)).cuda()
    for p in torch.FloatTensor(np.array(positions)).cuda():
        diff = (((possible_positions - p.unsqueeze(0)) ** 2).sum(-1))
        possible_positions = possible_positions[diff > dist]

    possible_positions = possible_positions[:self.cfg_policy.candidate.num_candidates]
    if possible_positions.shape[0] == self.cfg_policy.candidate.num_candidates:
        return possible_positions.data.cpu().numpy()
    else:
        return get_locations_random(self, positions, seed, dist=dist -.1)


# get the uncertainty from occupancy and perspective ray information
def get_uncertainty(self, occ_fun, checks, ray_points, ray_masks, dists, chunk_size, position):

    ray_points = ray_points.view(-1, 3)
    ray_masks = ray_masks.view(-1)

    value = torch.zeros(ray_points.shape[0]).cuda()
    considered_points = ray_points[ray_masks] # only get occuapncy for positiosn inside object space

    # apply occupancy predictions over chunks
    num_splits = (considered_points.shape[0] // chunk_size ) +1
    all_values = []
    for i in range(num_splits):
        points = considered_points[i*chunk_size:i*chunk_size + chunk_size]
        if points.shape[0]> 0:
            all_values.append(occ_fun(points))
    all_values = torch.cat(all_values)
    all_values = torch.sigmoid(all_values)


    value[ray_masks] = all_values
    value[~ray_masks] = 0
    value = value.view(checks, -1, 128)
    dists = dists.view(checks, -1, 128 )

    # compute direcitonal derivative
    if self.cfg_policy.uncert.dir_dir[position] > 0:
        dir = torch.abs(value[:, :, :-2] - value[:, :, 2:])
        dir_value = (1 - ((dir + 1e-8) ** self.cfg_policy.uncert.dir_dir[position]))
        dir_value = dir_value.view(checks, -1, 128-2)
    else:
        dir_value = 1

    # eliminate outer values to be same size as rate of change value
    occupancy = value[:,:,1:-1]
    dists = dists[:,:, 1:-1]

    # parameters for uncertainty
    pow_acc = self.cfg_policy.uncert.pow_acc
    pow_uncert = self.cfg_policy.uncert.pow_uncert
    lam = self.cfg_policy.uncert.uncert_lam
    sil_pow = self.cfg_policy.uncert.sil_pow

    # occupancy uncertainty
    c = torch.abs(0.5 - occupancy) * 2
    u = 1 - ((c + 1e-8) ** pow_uncert)

    # weight for relative size of samples in integral
    forward_value = torch.argmin(dists[0, :, 0])
    weight = (dists[0, forward_value, :]) ** 2
    weight = weight / weight.max()
    weight = weight.view(1, 1, -1)

    # uncertainty accumulation
    Tu =  (torch.relu((occupancy - .5) * 2) + 1e-10) ** pow_acc
    Tu = cumprod_exclusive(1 - Tu + 1e-10)

    # depth uncertainty
    uncert_depth = (u * weight * Tu * dir_value).sum(-1)

    # sil uncertainty
    pred_sil = cumprod_exclusive(1.0 - occupancy + 1e-10)[..., -1]
    uncert_sil = 1 - (torch.abs(0.5 - pred_sil) * 2).pow(sil_pow)

    # final uncert
    uncertainties =  (uncert_sil + lam) * uncert_depth

    return uncertainties

# volume rendering
def nerf_rendering(values, mask):
    bs = values.shape[0]
    occupancy = values.view(-1,128, 4)[:,:, 0]
    occupancy = torch.sigmoid(occupancy)
    alpha = occupancy * mask.view(occupancy.shape)

    colour =  torch.sigmoid(values[..., 1:]) * mask.unsqueeze(-1)
    colour = colour.view(-1, 128, 3)
    Ti = cumprod_exclusive(1.0 - alpha + 1e-10)

    weights = alpha * Ti
    rgb_map = weights[..., None] * colour
    rgb_map = rgb_map.sum(dim=-2)
    acc_map = weights.sum(dim=-1)
    rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map.view(bs, -1, 3)


def cumprod_exclusive(tensor):
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod


 # get intersection over union
def get_iou(pred_values, gt_values, threshold=0.5, apply_threshold = False ):

    if apply_threshold:
        pred_values[pred_values > threshold] = 1
        pred_values[pred_values <= threshold] = 0

    intersection = pred_values * gt_values
    union = pred_values + gt_values - intersection
    iou = 0
    for i, u in zip(intersection, union):
        iou += i.sum() / u.sum()
    iou = iou /float(pred_values.shape[0])
    return iou


# positional encoding from nerf paper
def positional_encoding( tensor, num_encoding_functions=10, include_input=True, log_sampling=True):
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)





