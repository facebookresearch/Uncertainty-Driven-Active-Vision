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
from torch.utils.tensorboard import SummaryWriter
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
from submitit.helpers import Checkpointable


import sys
sys.path.insert(0, "../")
from utils import CfgNode, train_utils, rendering, rays
from models import occupancy


class Engine(Checkpointable):
    def __init__(self, cfg, cfg_policy):
        self.cfg = cfg
        self.cfg_policy = cfg_policy

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

    def __call__(self) -> float:

        # get datasert
        self.get_loaders()

        # make model
        self.model = occupancy.model(self.cfg)
        self.model.cuda()
        self.load()

        # set renderer
        self.renderer = rendering.Renderer([128, 128])

        # apply the policy
        self.NBV()

    def get_loaders(self):
        set_type = "test" if self.cfg.eval else "valid"
        self.valid_data = data_loader.data(self.cfg, set_type=set_type)

    def NBV(self):
        self.random_initializations = self.cfg_policy.eval.num_initialization
        self.model.eval()

        total_iou = []
        # loop over the object
        for i in tqdm(range(len(self.valid_data.object_names))):

            # load the object
            obj = self.valid_data.object_names[i][0]
            self.object_name = self.valid_data.obj_location + f"{obj}.obj"
            mesh = trimesh.load(self.object_name)
            self.renderer.remove_objects()
            self.renderer.add_object(mesh)

            # get ground truth occupancies
            points, values = self.valid_data.get_voxels(
                obj, num_points=self.cfg_policy.eval.num_points_iou
            )
            self.points = torch.FloatTensor(points).cuda().contiguous().view(1, -1, 3)
            self.values = torch.FloatTensor(values).cuda().contiguous().view(1, -1)

            obj_iou = []
            # loop over the number of random initializations
            for r in range(self.random_initializations):
                seed = (
                    (i + 1)
                    * self.random_initializations
                    * self.cfg.NBV.budget
                    * (r + 1)
                )
                obj_iou.append(self.eval_initialization(seed))

            # logging
            total_iou.append(obj_iou)
            for b in range(self.cfg.NBV.budget):
                cur_iou = np.array(total_iou).reshape(-1, self.cfg.NBV.budget)
                cur_iou = (cur_iou[:, b]).mean()
                self.writer.add_scalar(f"policy_eval/iou_{b}", cur_iou, i)
            message = f"Validation ||  IoU: {cur_iou :.5f} "
            tqdm.write(message)

        ious = torch.FloatTensor(np.array(total_iou))
        # get mean of policy
        iou_mean = ious.view(-1, self.cfg.NBV.budget)

        iou_stds = []
        # get std of policy
        for b in range(self.cfg.NBV.budget):
            stds = ious[:, :, b].std(1).mean().item()
            iou_stds.append(stds)

        # get worst of policy over initializatiouns
        iou_worst = ious.min(1)[0].mean(0)

        # report policy perfromance
        print("iou: ", iou_mean.mean(0).data.cpu().numpy())
        print("iou std: ", iou_stds)
        print("iou worst : ", iou_worst.data.cpu().numpy())

    # loop over numer of images in budget
    def eval_initialization(self, seed):
        imgs = torch.zeros(1, self.cfg.NBV.budget, 3, 128, 128).cuda()
        mats = torch.zeros(1, self.cfg.NBV.budget, 3, 4).cuda()
        params = torch.zeros(1, self.cfg.NBV.budget, 7).cuda()
        positions = []
        policy_iou = []
        for j in range(1, self.cfg.NBV.budget + 1):
            self.seed = seed + j
            imgs, mats, params, positions, iou = self.eval_policy(
                imgs, mats, params, positions, j
            )
            policy_iou.append(iou)
        return policy_iou

    # pass values to the correct policy
    def eval_policy(self, imgs, mats, params, positions, position):

        # using the random policy
        if self.cfg_policy.NBV.policy == "random" or position <= 1:
            with torch.no_grad():
                return self.random_policy(imgs, mats, params, positions, position)

        # using the even or odd policies
        elif  "even" in self.cfg_policy.NBV.policy or "odd" in self.cfg_policy.NBV.policy:
            with torch.no_grad():
                return self.even_and_odd_policy(imgs, mats, params, positions, position)

        # using the candidate policy
        elif self.cfg_policy.NBV.policy == "candidate":
            with torch.no_grad():
                return self.candidate_policy(imgs, mats, params, positions, position)

        # using the gradient policy
        elif self.cfg_policy.NBV.policy == "gradient":
            return self.gradient_policy(imgs, mats, params, positions, position)

    # evaluate the perfromance of the chosen image
    def eval_setting(self, imgs, mats, params, position):
        with torch.no_grad():
            split_points = self.split_values(
                self.points, self.cfg_policy.eval.chunk_size
            )
            value = []
            for sp in split_points:
                v = torch.sigmoid(self.model(imgs, sp, mats, params, position))
                value.append(v)
            pred_values = torch.cat(value, dim=1)
            loss_iou = train_utils.get_iou(
                pred_values, self.values, apply_threshold=True
            )
            return loss_iou.mean().item()

    # the random policy
    def random_policy(self, imgs, mats, params, positions, position):

        # set a set random camera position and orientation
        location = train_utils.get_locations_random(self, positions, self.seed)[0]
        orientation = self.renderer.cam_from_positions(location)
        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    # the even or odd policy
    def even_and_odd_policy(self, imgs, mats, params, positions, position):

        # select new positions furthest from current views
        location = train_utils.get_location_even(self, positions, self.seed)
        orientation = self.renderer.cam_from_positions(location)

        # select every second even position if using odd policy
        if "odd" in self.cfg_policy.NBV.policy:
            positions.append(location)
            location = train_utils.get_location_even(self, positions, self.seed)
            orientation = self.renderer.cam_from_positions(location)

        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    # the candidate policy
    def candidate_policy(self, imgs, mats, params, positions, position):

        # get image embedding and define function from embedding to occupancy
        embedding = self.model.image_to_embedding(imgs, position - 1)
        occ_fun = lambda x: self.model.embedding_to_values(
            embedding, x.unsqueeze(0), mats, params, position - 1
        )[0]

        # set random postions to consider
        locations = train_utils.get_locations_random(
            self, positions, self.seed, dist=self.cfg_policy.NBV.location_dist
        )
        locations = torch.FloatTensor(locations).cuda()

        # get ray info for considered perspectives
        ray_points, ray_masks, dists = rays.get_rays(
            self, locations, resolution=self.cfg_policy.candidate.resolution
        )

        # get uncertainty of each ray
        uncertainties = train_utils.get_uncertainty(
            self,
            occ_fun,
            self.cfg_policy.candidate.num_candidates,
            ray_points,
            ray_masks,
            dists,
            self.cfg_policy.eval.chunk_size,
            position - 2,
        )
        # get the highest uncertainty position
        best_position = uncertainties.mean(-1).argmax()
        location = locations[best_position].data.cpu().numpy()
        orientation = self.renderer.cam_from_positions(location)

        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    # the gradient policy
    def gradient_policy(self, imgs, mats, params, positions, position):

        # get random position
        location = train_utils.get_locations_random(
            self, positions, self.seed, dist=self.cfg_policy.NBV.location_dist
        )[0]
        location = torch.FloatTensor(location).cuda()

        # get image embedding
        with torch.no_grad():
            embedding = self.model.image_to_embedding(imgs, position - 1)

        location = torch.nn.Parameter(location.clone())
        optimizer = optim.Adam([location], lr=self.cfg_policy.grad.lr)
        cur_positions = torch.FloatTensor(np.array(positions)).cuda()

        # gradient update
        for i in range(self.cfg_policy.grad.steps):
            optimizer.zero_grad()
            # place loaction in the search space
            norm_locations = (
                torch.nn.functional.normalize(location.unsqueeze(0), dim=-1) * 0.8
            )

            # fucntion from embedding to occupancy
            occ_fun = lambda x: self.model.embedding_to_values(
                embedding, x.unsqueeze(0), mats, params, position - 1
            )[0]

            # get rays for perspective
            ray_points, ray_masks, dists = rays.get_rays(
                self, norm_locations.clone(), resolution=self.cfg_policy.grad.resolution
            )

            # get uncertainty
            uncertainties = train_utils.get_uncertainty(
                self, occ_fun, 1, ray_points, ray_masks, dists, 20000, position - 2
            )
            uncertainties = uncertainties.mean()

            if i == 0:
                best_uncertainty = uncertainties.item()
                best_location = norm_locations.clone()[0].data.cpu().numpy()

            if uncertainties > best_uncertainty:
                best_uncertainty = uncertainties.item()
                best_location = norm_locations.clone()[0].data.cpu().numpy()

            # loss for gradient
            loss = -uncertainties
            # distance regularizers to existsing camera positions
            distance = torch.sqrt(
                ((norm_locations.unsqueeze(0) - cur_positions.unsqueeze(1)) ** 2).sum(
                    -1
                )
            )
            distance = torch.min(distance, dim=0)[0]
            dist_reg = -self.cfg_policy.grad.dist_reg * distance.mean()

            # regulizer to defined search space
            norm_distance = self.cfg_policy.grad.norm_reg * torch.abs(
                torch.sqrt((location ** 2).sum()) - 0.8
            )

            loss += dist_reg + norm_distance

            # grad step
            loss.backward()
            optimizer.step()

        location = best_location
        orientation = self.renderer.cam_from_positions(location)

        return self.update_and_eval(
            location, orientation, imgs, mats, params, positions, position
        )

    #  update the input for the chosen view and evaluate
    def update_and_eval(
        self, location, orientation, imgs, mats, params, positions, position
    ):
        # set image from this camera
        self.renderer.update_camera_pose(location, orientation)
        colour = self.valid_data.preprocess(Image.fromarray(self.renderer.render()))

        # update inputs with this information
        matrix = self.valid_data.matrix_from_params(location, orientation)
        quaternion = R.from_euler("xyz", orientation, degrees=True).as_quat()
        param = torch.FloatTensor(np.concatenate((location, quaternion)))
        imgs[0, position - 1] = colour.cuda()
        mats[0, position - 1] = matrix.cuda()
        params[0, position - 1] = param.cuda()
        positions.append(location)

        return (
            imgs,
            mats,
            params,
            positions,
            self.eval_setting(imgs, mats, params, position),
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

    # load the model
    def load(self):
        check_point = self.savedir + "_iou_best.pt"
        check_point = torch.load(check_point)
        self.model.load_state_dict(check_point["model"], strict=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy_config",
        type=str,
        default="../configs/ABC_3D_NBV.yml",
        help="Path to (.yml) config file.",
    )
    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="use less data, for debugging.",
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="evaluate the policy"
    )

    configargs = parser.parse_args()

    # load the policy config file
    with open(configargs.policy_config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_policy = CfgNode(cfg_dict)

    # load the recon model policy
    with open(cfg_policy.base.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    cfg.limit_data = configargs.limit_data
    cfg.eval = configargs.eval
    cfg.NBV.budget = cfg_policy.NBV.budget

    trainer = Engine(cfg, cfg_policy)
    trainer()
