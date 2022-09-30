# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
from PIL import Image


IMAGE_LOCATION = "../data/images/"
VOXEL_LOCATION = "../data/voxels/"
OBJ_LOCATION = "../data/objects/"
NUM_IMAGES = 25


class data(object):
    def __init__(self, args, set_type="train"):
        # initialization of data locations
        self.args = args
        self.set_type = set_type
        object_names = np.load("../utils/obj_names.npy")
        self.obj_location = OBJ_LOCATION
        self.training = set_type == "train"
        self.object_names = []

        self.preprocess = transforms.Compose([transforms.ToTensor()])

        # set seed
        random.seed(1)
        random.shuffle(object_names)
        if args.limit_data:
            object_names = object_names[:1000]

        for i, n in enumerate(tqdm(object_names)):
            if os.path.exists(VOXEL_LOCATION + n + ".npy"):
                if set_type == "train" and int(n) < 23000:
                    self.object_names.append([n, None])
                if set_type == "valid" and int(n) >= 23000 and int(n) < 24500:
                    self.object_names.append([n, i])
                if set_type == "test" and int(n) >= 24500:
                    self.object_names.append([n, i])
        print(f"The number of {set_type} set objects found : {len(self.object_names)}")

    def __len__(self):
        return len(self.object_names)

    # get project matrix from position and orientation
    def matrix_from_params(self, position, orientation):
        orientation = R.from_euler("xyz", orientation, degrees=True).as_matrix()

        R_bcam2cv = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

        R_world2bcam = orientation.transpose()
        T_world2bcam = -1 * R_world2bcam @ position

        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam

        # put into 3x4 matrix
        RT = np.array(
            (
                list(R_world2cv[0][:]) + [T_world2cv[0]],
                list(R_world2cv[1][:]) + [T_world2cv[1]],
                list(R_world2cv[2][:]) + [T_world2cv[2]],
            )
        )

        f = 221.7025
        K = np.array([[f, 0, 128.0], [0, f, 128.0], [0, 0, 1]])

        return torch.FloatTensor(K.dot(RT))

    # get images from dataset
    def get_img(self, obj, seed):
        img_location = IMAGE_LOCATION + f"/{obj}/"
        if seed is not None:
            imgs_nums = np.random.RandomState(seed=seed).choice(
                np.arange(NUM_IMAGES), self.args.NBV.budget + 1, replace=False
            )
        else:
            imgs_nums = np.random.choice(
                np.arange(NUM_IMAGES), self.args.NBV.budget + 1, replace=False
            )
        imgs = []
        matricies = []
        cam_params = []

        for n in imgs_nums:
            # load image
            input_image = Image.open(img_location + f"{n}.png")
            input_tensor = self.preprocess(input_image)
            imgs.append(input_tensor)
            # load image parameters
            params = np.load(img_location + f"P_{n}.npy", allow_pickle=True).item()
            position = params["position"]
            rotation = params["rotation"]
            matrix = self.matrix_from_params(position, rotation)
            matricies.append(matrix)
            quaternion = R.from_euler("xyz", rotation, degrees=True).as_quat()
            cam_params.append(torch.FloatTensor(np.concatenate((position, quaternion))))

        imgs = torch.stack(imgs)
        matricies = torch.stack(matricies)
        cam_params = torch.stack(cam_params)
        return imgs, matricies, cam_params

    # load occupancy information
    def get_voxels(self, obj, seed, num_points):

        if seed is not None:
            local_positions = np.random.RandomState(seed=seed).uniform(
                -0.4, 0.4, (num_points, 3)
            )
        else:
            local_positions = np.random.uniform(-0.4, 0.4, (num_points, 3))

        points = np.load(VOXEL_LOCATION + f"{obj}.npy")
        dim = 128
        voxels = np.zeros((dim, dim, dim))
        voxels[tuple(points)] = 1
        absolute_positions = np.array(local_positions)
        absolute_positions = ((absolute_positions + 0.4) / 0.8) * (dim - 1) // 1
        xs, ys, zs = np.clip(absolute_positions, 0, dim - 1).astype(int).transpose()
        values = voxels[(xs, ys, zs)]

        return local_positions, values

    def __getitem__(self, index):
        object_name, seed = self.object_names[index]

        # load image
        imgs, matricies, params = self.get_img(object_name, seed)

        # get voxels
        if not self.training:
            num_points = 100000
            positions, values = self.get_voxels(object_name, seed, num_points)
            values = torch.FloatTensor(values)
            positions = torch.FloatTensor(positions)
        else:
            positions, values = None, None

        data = {
            "imgs": imgs,
            "names": object_name,
            "matricies": matricies,
            "params": params,
            "positions": positions,
            "gt_values": values,
        }
        return data

    def collate(self, batch):
        data = {}
        data["names"] = [item["names"] for item in batch]
        data["imgs"] = torch.cat([item["imgs"].unsqueeze(0) for item in batch])
        data["matricies"] = torch.cat(
            [item["matricies"].unsqueeze(0) for item in batch]
        )
        data["params"] = torch.cat([item["params"].unsqueeze(0) for item in batch])
        if not self.training:
            data["positions"] = torch.cat(
                [item["positions"].unsqueeze(0) for item in batch]
            )
            data["gt_values"] = torch.cat(
                [item["gt_values"].unsqueeze(0) for item in batch]
            )

        return data
