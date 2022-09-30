# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from tqdm import tqdm, trange
import torch.optim as optim
import argparse
import yaml
import data_loader
from submitit.helpers import Checkpointable
from torch.utils.tensorboard import SummaryWriter
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
import math


import sys
sys.path.insert(0, "../")
from utils import CfgNode, train_utils, rendering, rays
from models import occupancy_and_colour


class Engine(Checkpointable):
    def __init__(self, cfg, cfg_policy):
        self.cfg = cfg
        self.cfg_policy = cfg_policy

        # set seed
        seed = self.cfg.experiment.randomseed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Setup logging.
        logdir = os.path.join(
            "logs",
            str(self.cfg_policy.NBV.exp_name),
            str(self.cfg_policy.NBV.inst_name),
        )
        os.makedirs(logdir, exist_ok=True)
        self.writer = SummaryWriter(logdir)
        self.savedir = os.path.join(
            "save_point",
            str(self.cfg.experiment.experiment_id),
            str(self.cfg.experiment.id),
        )
        self.loss_names = ["psnr", "IoI"]

    def __call__(self):

        # get datasert
        self.get_loaders()

        # make model
        self.model = occupancy_and_colour.model(self.cfg)
        self.model.cuda()
        self.load()

        self.renderer = rendering.Renderer([128, 128])
        self.NBV()

    def get_loaders(self):
        # evaluation dataloder
        set_type = "test" if self.cfg.eval else "valid"
        self.valid_data = data_loader.data(self.cfg, set_type=set_type)

    # ground truth image values for eval
    def get_gt_colours(self, seed):
        gt_values, ray_points, ray_masks, gt_params = [], [], [], []
        for k in range(self.cfg_policy.eval.num_imgs):

            # get random image
            gt_img, gt_mat, gt_param = self.get_random_img(seed * k + k)
            # get ray samples for image
            ray_p, ray_m, _ = rays.get_rays(self, gt_param[:3].unsqueeze(0))
            # subsample rays
            gt_v, ray_p, ray_m = rays.subsample_rays(
                self,
                ray_p,
                ray_m,
                gt_img.unsqueeze(0),
                num_points=self.cfg_policy.eval.num_rays_per_img,
                seed=seed * k + k,
            )
            gt_values.append(gt_v)
            ray_points.append(ray_p)
            ray_masks.append(ray_m)
            gt_params.append(gt_param)
        gt_values = torch.cat(gt_values)
        ray_points = torch.cat(ray_points)
        ray_masks = torch.cat(ray_masks)
        gt_params = torch.stack(gt_params)
        self.gt = [gt_values, ray_points, ray_masks, gt_params]

    # ground truth occupancy values for eval
    def get_gt_occ(self, object_name, i):
        positions, voxels = self.valid_data.get_voxels(
            object_name, i, self.cfg_policy.eval.num_points_iou
        )
        voxels = torch.FloatTensor(voxels).cuda().view(-1)
        positions = torch.FloatTensor(positions).cuda().view(-1, 3)
        self.voxels = [voxels, positions]

    def NBV(self):
        random_initializations = self.cfg_policy.eval.num_initialization
        self.model.eval()

        total_loss = []
        for i in tqdm(range(len(self.valid_data.object_names))):

            # load the object
            obj = self.valid_data.object_names[i][0]
            self.object_name = self.valid_data.obj_location + f"{obj}.obj"
            mesh = trimesh.load(self.object_name)
            self.renderer.remove_objects()
            self.renderer.add_object(mesh)

            # get gt info
            self.get_gt_occ(obj, i)
            self.get_gt_colours(i)

            obj_loss = []
            for r in range(random_initializations):
                seed = (i + 1) * random_initializations * self.cfg.NBV.budget * (r + 1)
                obj_loss.append(self.eval_initialization(seed))
            total_loss.append(obj_loss)

            # iterate over psnr and iou
            for e, l in enumerate(self.loss_names):
                # interate over number of input images
                message = f"{l} || "
                for b in range(self.cfg.NBV.budget):
                    cur_loss = np.array(total_loss)[..., e].reshape(
                        -1, self.cfg.NBV.budget
                    )
                    cur_loss = (cur_loss[:, b]).mean()
                    self.writer.add_scalar(f"policy_eval/{l}_{b}", cur_loss, i)
                    message += f"{b} : {cur_loss :.5f}  "
                tqdm.write(message)
            tqdm.write("-" * 20)

        for e, l in enumerate(self.loss_names):
            losses = np.array(
                [np.array(total_loss)[..., i, e] for i in range(self.cfg.NBV.budget)]
            )
            losses = losses.transpose(1, 2, 0)
            mean = losses.reshape(-1, self.cfg.NBV.budget).mean(0)

            print(f"{l} mean : ", mean)

            min = losses.min(1).mean(0)
            print(f"{l} min : ", min)

            stds = []
            for b in range(self.cfg.NBV.budget):
                std = losses[:, :, b].std(1).mean().item()
                stds.append(std)

            print(f"{l} stds : ", stds)

    # evaluate an the object with current initialization
    def eval_initialization(self, seed):
        imgs = torch.zeros(1, self.cfg.NBV.budget, 3, 128, 128).cuda()
        mats = torch.zeros(1, self.cfg.NBV.budget, 3, 4).cuda()
        params = torch.zeros(1, self.cfg.NBV.budget, 7).cuda()
        positions = []
        policy_loss = []
        for j in range(1, self.cfg.NBV.budget + 1):
            seed = self.seed = seed + j
            imgs, mats, params, positions, loss = self.eval_policy(
                imgs, mats, params, positions, j, seed
            )
            policy_loss.append(loss)
        return policy_loss

    # eval the current inputs
    def eval_setting(self, imgs, mats, params, position):
        with torch.no_grad():

            # get embedding for the input images
            embedding = self.model.image_to_embedding(imgs, position)

            # iterate over perspectives to get PSNR
            colours = []
            col_fun = lambda x, y: self.model.embedding_to_values(
                embedding, x, mats, params, y, position
            )[0]
            for i in range(self.gt[1].shape[0]):
                pred_values = torch.zeros((self.gt[1][i].shape[0], 4)).cuda()
                considered_points = self.gt[1][i][self.gt[2][i]].unsqueeze(0)
                split_points = self.split_values(
                    considered_points, self.cfg_policy.eval.chunk_size
                )
                all_values = []
                for sp in split_points:
                    if sp.view(-1).shape[0] == 0:
                        continue
                    all_values.append(col_fun(sp, self.gt[-1][i].unsqueeze(0)))
                all_values = torch.cat(all_values)
                pred_values[self.gt[2][i]] = all_values
                colour = train_utils.nerf_rendering(
                    pred_values.unsqueeze(0), self.gt[2][i]
                )
                colours.append(colour)

            colours = torch.cat(colours)
            mse = torch.nn.MSELoss()(colours, self.gt[0])
            psnr = 10 * math.log10(1 / mse.item())
            if psnr == 0 or math.isnan(psnr) or math.isinf(psnr):
                psnr = 5.0

            # get iou
            fun = lambda x: torch.sigmoid(
                self.model.embedding_to_occ(embedding, x, mats, params, position)
            )
            pos = self.voxels[1].unsqueeze(0)
            split_points = self.split_values(pos, self.cfg_policy.eval.chunk_size)
            all_values = []
            for sp in split_points:
                if sp.view(-1).shape[0] == 0:
                    continue
                all_values.append(fun(sp))
            pred_voxels = torch.cat(all_values).view(1, -1)
            iou = (
                train_utils.get_iou(
                    pred_voxels,
                    self.voxels[0][None],
                    threshold=0.4,
                    apply_threshold=True,
                )
                .mean()
                .item()
            )
            if iou == 0 or math.isnan(iou) or math.isinf(iou):
                iou = 0.01
            loss = [psnr, iou]

        return loss

    # pass values to the correct policy
    def eval_policy(self, imgs, mats, params, positions, position, seed):

        # using the random policy
        if self.cfg_policy.NBV.policy == "random" or position <= 1:
            with torch.no_grad():
                return self.random_policy(imgs, mats, params, positions, position, seed)

        # using the even or odd policy
        elif (
            "even" in self.cfg_policy.NBV.policy or "odd" in self.cfg_policy.NBV.policy
        ):
            with torch.no_grad():
                return self.even_or_odd_policy(
                    imgs, mats, params, positions, position, seed
                )

        # using the candidate policy
        elif self.cfg_policy.NBV.policy == "candidate":
            with torch.no_grad():
                return self.candidate_policy(
                    imgs, mats, params, positions, position, seed
                )

        # using the gradient policy
        elif self.cfg_policy.NBV.policy == "gradient":
            return self.gradient_policy(imgs, mats, params, positions, position, seed)

    #  update the input for the chosen view and evaluate
    def update_and_eval(
        self, location, orientation, imgs, mats, params, positions, position
    ):
        # render image from chosen perspective
        self.renderer.update_camera_pose(location, orientation)
        colour = self.valid_data.preprocess(Image.fromarray(self.renderer.render()))

        # additional inputs
        matrix = self.valid_data.matrix_from_params(location, orientation)
        quaternion = R.from_euler("xyz", orientation, degrees=True).as_quat()
        param = torch.FloatTensor(np.concatenate((location, quaternion)))

        # update inputs
        positions.append(location)
        imgs[0, position - 1] = colour.cuda()
        mats[0, position - 1] = matrix.cuda()
        params[0, position - 1] = param.cuda()

        return (
            imgs,
            mats,
            params,
            positions,
            self.eval_setting(imgs, mats, params, position),
        )

    # the random policy
    def random_policy(self, imgs, mats, params, positions, position, seed):
        location = self.renderer.random_position(0.8, seed=seed)
        orientation = self.renderer.cam_from_positions(location)
        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    # the odd or even policies
    def even_or_odd_policy(self, imgs, mats, params, positions, position, seed):
        location = train_utils.get_location_even(self, positions, seed)
        orientation = self.renderer.cam_from_positions(location)

        if "odd" in self.cfg_policy.NBV.policy:
            positions.append(location)
            location = train_utils.get_location_even(self, positions, seed)
            orientation = self.renderer.cam_from_positions(location)

        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    def candidate_policy(self, imgs, mats, params, positions, position, seed=None):
        # get embedding for current images
        embedding = self.model.image_to_embedding(imgs, position - 1)

        # define function from
        occ_fun = lambda x: self.model.embedding_to_occ(
            embedding, x.unsqueeze(0), mats, params, position - 1
        )[0, ..., 0]
        num_candidates = self.cfg_policy.candidate.num_candidates

        # set random postions to consider
        locations = train_utils.get_locations_random(
            self, positions, self.seed, dist=self.cfg_policy.NBV.location_dist
        )
        locations = torch.FloatTensor(locations).cuda()

        # get uncertainty of each perspective
        ray_points, ray_masks, dists = rays.get_rays(
            self, locations, resolution=self.cfg_policy.candidate.resolution
        )
        uncertainties = train_utils.get_uncertainty(
            self,
            occ_fun,
            num_candidates,
            ray_points,
            ray_masks,
            dists,
            self.cfg_policy.eval.chunk_size,
            position - 2,
        )

        # select perspective with highest uncertainty
        best_position = uncertainties.mean(-1).argmax()
        location = locations[best_position].data.cpu().numpy()
        orientation = self.renderer.cam_from_positions(location)

        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    def gradient_policy(self, imgs, mats, params, positions, position, seed=None):

        # selection initial position
        location = torch.FloatTensor(
            self.renderer.random_position(0.8, seed=seed)
        ).cuda()

        # embed current images
        with torch.no_grad():
            embedding = self.model.image_to_embedding(imgs, position - 1)

        # set optimization parameters
        locations = torch.nn.Parameter(location.clone())
        optimizer = optim.Adam([locations], lr=self.cfg_policy.grad.lr)
        cur_positions = torch.FloatTensor(np.array(positions)).cuda()

        # grad update
        for i in range(self.cfg_policy.grad.steps):
            optimizer.zero_grad()
            # place location back into search space
            norm_locations = (
                torch.nn.functional.normalize(locations.unsqueeze(0), dim=-1) * 0.8
            )

            # get uncertainty
            ray_points, ray_masks, dists = rays.get_rays(
                self, norm_locations.clone(), resolution=self.cfg_policy.grad.resolution
            )
            occ_fun = lambda x: self.model.embedding_to_occ(
                embedding, x.unsqueeze(0), mats, params, position - 1
            )[0, ..., 0]

            uncertainties = train_utils.get_uncertainty(
                self,
                occ_fun,
                1,
                ray_points,
                ray_masks,
                dists,
                self.cfg_policy.eval.chunk_size,
                position - 2,
            )
            uncertainties = uncertainties.mean()

            if i == 0:
                best_uncertainty = uncertainties.item()
                best_location = norm_locations.clone()[0]

            if uncertainties > best_uncertainty or i == 0:
                best_uncertainty = uncertainties.item()
                best_location = norm_locations.clone()[0]

            # defined loss
            loss = -uncertainties

            # distance regularizer
            distance = torch.sqrt(
                ((norm_locations.unsqueeze(0) - cur_positions.unsqueeze(1)) ** 2).sum(
                    -1
                )
            )
            distance = torch.min(distance, dim=0)[0]
            dist_reg = -self.cfg_policy.grad.dist_reg * distance.mean()

            # regulizer to defined search space
            norm_distance = self.cfg_policy.grad.norm_reg * torch.abs(
                torch.sqrt((locations ** 2).sum()) - 0.8
            )

            loss += dist_reg + norm_distance
            loss.backward()
            optimizer.step()

        location = best_location.data.cpu().numpy()
        orientation = self.renderer.cam_from_positions(location)
        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    # split the input points into chunks so we dont run out of memeory
    def split_values(self, inputs, num_points):
        values = []
        length = inputs.shape[1] // num_points
        for i in range(length):
            values.append(inputs[:, i * num_points : i * num_points + num_points])
        values.append(inputs[:, (num_points) * length :])
        if values[-1].shape[1] == 0:
            values = values[:-1]
        total = 0
        for v in values:
            total += v.shape[1]
        assert total == inputs.shape[1]
        return values

    def load(self):
        check_point = self.savedir + "_loss_best.pt"
        check_point = torch.load(check_point)
        self.model.load_state_dict(check_point["model"], strict=False)

    # render random image of the object
    def get_random_img(self, seed, position=None):
        if position is None:
            start_location = self.renderer.random_position(0.8, seed=seed)
        else:
            start_location = position
        start_orientation = self.renderer.cam_from_positions(start_location)
        self.renderer.update_camera_pose(start_location, start_orientation)
        colour = self.valid_data.preprocess(Image.fromarray(self.renderer.render()))
        matrix = self.valid_data.matrix_from_params(start_location, start_orientation)
        quaternion = R.from_euler("xyz", start_orientation, degrees=True).as_quat()
        param = torch.FloatTensor(np.concatenate((start_location, quaternion)))
        return colour.cuda(), matrix.cuda(), param.cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy_config",
        type=str,
        default="../configs/ABC_2D_NBV.yml",
        help="Path to (.yml) config file.",
    )
    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="use less data, for debugging.",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="evaluate")

    configargs = parser.parse_args()

    # config for the policy
    with open(configargs.policy_config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_policy = CfgNode(cfg_dict)

    # config for the loaded recon model
    with open(cfg_policy.base.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    cfg.limit_data = configargs.limit_data
    cfg.eval = configargs.eval
    cfg.NBV.budget = cfg_policy.NBV.budget

    trainer = Engine(cfg, cfg_policy)
    trainer()
