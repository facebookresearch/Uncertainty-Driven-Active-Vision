# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import numpy as np
from tqdm import tqdm, trange
import torch.optim as optim
import argparse
import yaml
import data_loader
from submitit.helpers import Checkpointable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import skimage
from einops import rearrange
from PIL import Image
import scipy

sys.path.insert(0, "../")
from utils import CfgNode, train_utils, rendering
from models import occupancy


class Engine(Checkpointable):
    def __init__(self, cfg):
        self.cfg = cfg

        # initial settings
        self.best_loss_bce = 1000
        self.best_loss_iou = 0
        self.epoch = 0
        self.renderer = rendering.Renderer()
        self.last_improvement = 0

        # set seed
        seed = self.cfg.experiment.randomseed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Setup logging.
        logdir = os.path.join(
            "logs", str(self.cfg.experiment.experiment_id), str(self.cfg.experiment.id)
        )
        os.makedirs(logdir, exist_ok=True)
        self.savedir = os.path.join(
            "save_point",
            str(self.cfg.experiment.experiment_id),
            str(self.cfg.experiment.id),
        )
        os.makedirs(self.savedir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    def __call__(self) -> float:

        # get dataset
        self.get_loaders()

        # make model
        self.model = occupancy.model(self.cfg)
        self.model.cuda()

        # optimizer
        self.model.cuda()
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.cfg.optimizer.lr, weight_decay=0)

        # evaluate the trained model
        if self.cfg.eval:
            self.load(best=True)
            self.validate()
            self.eval_render()
            exit()

        # resume training
        if not cfg.reset:
            self.resume()

        # start training
        start_iter = self.epoch
        for i in range(start_iter, cfg.experiment.num_epochs):
            self.train()
            if i > start_iter and i % 2 == 0:
                self.validate()
                if self.check_values():
                    return
                self.render()
                self.save(best=False)
            self.epoch += 1

    def get_loaders(self):

        # training dataloader
        train_data = data_loader.data(self.cfg, set_type="train")
        train_data[0]  # check loader works

        train_loader = DataLoader(
            train_data,
            batch_size=self.cfg.experiment.batch_size,
            shuffle=True,
            num_workers=10,
            collate_fn=train_data.collate,
        )
        self.train_loader = train_loader
        self.train_data = train_data

        # evaluation dataloder
        set_type = "test" if self.cfg.eval else "valid"
        valid_data = data_loader.data(self.cfg, set_type=set_type)
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.cfg.experiment.batch_size,
            shuffle=False,
            num_workers=10,
            collate_fn=valid_data.collate,
        )
        self.valid_loader = valid_loader
        self.valid_data = valid_data

    # resume training
    def resume(self):
        try:
            self.load(best=False)
            message = f"Loaded at epoch {self.epoch}"
            print("*" * len(message))
            print(message)
            print("*" * len(message))
        except:
            message = "Was not able to load "
            print("*" * len(message))
            print(message)
            print("*" * len(message))

    def train(self):
        self.model.train()
        average_bce = []
        average_iou = []

        for k, batch in enumerate(tqdm(self.train_loader, smoothing=0)):
            self.optimizer.zero_grad()

            # inputs
            positions = batch["positions"].cuda()
            gt_values = batch["gt_values"].cuda()
            imgs = batch["imgs"].cuda()
            matricies = batch["matricies"].cuda()
            params = batch["params"].cuda()
            num_images = np.random.choice(
                np.arange(1, self.cfg.NBV.budget + 1)
            )  # set number of input images in the batch

            # inference
            pred_values = self.model(imgs, positions, matricies, params, num_images)

            # loss
            bce_loss = torch.nn.BCEWithLogitsLoss()(pred_values, gt_values)
            iou_loss = train_utils.get_iou(torch.sigmoid(pred_values), gt_values).mean()
            if torch.isnan(iou_loss):
                iou_loss = torch.FloatTensor([0]).sum()
            loss = self.cfg.loss.bce_loss * bce_loss - self.cfg.loss.iou_loss * iou_loss

            # optimization
            loss.backward()
            average_bce.append(bce_loss.item())
            average_iou.append(iou_loss.item())
            self.optimizer.step()

            # logging
            if k % 10 == 0 or self.cfg.limit_data:
                message = (
                    f"Train || Epoch: {self.epoch}, bce: {bce_loss.item():.5f} "
                    f"iou: {iou_loss.item():.5f}"
                )
                message += f"  || iou:  {self.best_loss_iou :.5f}, bce:  {self.best_loss_bce :.5f}"
                tqdm.write(message)

        # logging
        average_bce = np.array(average_bce).mean()
        average_iou = 1 - np.array(average_iou).mean()
        self.writer.add_scalar("train/bce", average_bce, self.epoch)
        self.writer.add_scalar("train/iou", average_iou, self.epoch)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            loss_total_bce = [0 for _ in range(self.cfg.NBV.budget)]
            loss_total_iou = [0 for _ in range(self.cfg.NBV.budget)]

            for k, batch in enumerate(tqdm(self.valid_loader, smoothing=0)):
                for j in range(1, self.cfg.NBV.budget + 1):

                    # inputs
                    positions = batch["positions"].cuda()
                    gt_values = batch["gt_values"].cuda()
                    imgs = batch["imgs"].cuda()
                    matricies = batch["matricies"].cuda()
                    params = batch["params"].cuda()

                    # inference
                    pred_values = []
                    for i in range(10):
                        pred_values.append(
                            self.model(
                                imgs,
                                positions[:, i * 10000 : i * 10000 + 10000],
                                matricies,
                                params,
                                j,
                            )
                        )
                    pred_values = torch.cat(pred_values, dim=1)

                    # loss
                    loss_BCE = torch.nn.BCEWithLogitsLoss()(pred_values, gt_values)
                    loss_iou = train_utils.get_iou(
                        torch.sigmoid(pred_values),
                        gt_values,
                        threshold=0.4,
                        apply_threshold=True,
                    ).mean()
                    if torch.isnan(loss_iou):
                        loss_iou = torch.FloatTensor([0]).sum()
                    loss_total_bce[j - 1] += loss_BCE.item()
                    loss_total_iou[j - 1] += loss_iou.item()

            # logging
            loss_total_bce = np.array(loss_total_bce) / float(k + 1)
            loss_total_iou = np.array(loss_total_iou) / float(k + 1)

            message = (
                f"Validation Total || Epoch: {self.epoch}, BCE: {loss_total_bce} "
                f"IoU: {loss_total_iou}  "
            )
            message += (
                f" || iou:  {self.best_loss_iou :.5f}, bce:  {self.best_loss_bce :.5f}"
            )
            tqdm.write(message)

            for j in range(self.cfg.NBV.budget):
                self.writer.add_scalar(
                    f"validation/bce_{j}", loss_total_bce[j], self.epoch
                )
                self.writer.add_scalar(
                    f"validation/iou_{j}", loss_total_iou[j], self.epoch
                )

            self.current_loss_bce = loss_total_bce[-1]
            cur_loss = np.array(loss_total_iou).mean()
            self.writer.add_scalar(f"validation/iou_mean", cur_loss, self.epoch)
            self.current_loss_iou = cur_loss

    # used to separate inference over chunks
    def split_values(self, inputs, splits):
        values = []
        length = inputs.shape[0] // splits
        for i in range(splits):
            values.append(inputs[i * length : i * length + length])
        values.append(inputs[(splits) * length :])

        total = 0
        for v in values:
            total += v.shape[0]
        assert total == inputs.shape[0]
        return values

    # render the predictions
    def render(self):
        print("\n" + "*" * 30)
        print(f"Rendering Images")
        print("*" * 30)
        self.model.eval()

        with torch.no_grad():
            seed = 0
            # loop over training and validation example
            for e, loader in enumerate([self.train_data, self.valid_data]):
                split = "train" if e == 0 else "valid"

                # basic info
                object_name, _ = loader.object_names[0]
                imgs, matricies, params = loader.get_img(object_name, seed)
                gt_voxels, positions, orig_positions = loader.get_all_voxels(
                    object_name
                )
                render_imgs = rearrange(imgs, "b c w h -> c w (b h)")
                self.writer.add_image(
                    f"{split}/input", render_imgs.data.cpu().numpy(), 0
                )

                # inputs
                imgs = imgs.cuda().unsqueeze(0)
                matricies = matricies.cuda().unsqueeze(0)
                params = params.cuda().unsqueeze(0)
                positions = torch.FloatTensor(positions).cuda()
                split_positions = self.split_values(positions, 50)

                # loop over numner of views
                for num_images in range(1, self.cfg.NBV.budget + 1):

                    # get values
                    pred_values = []
                    for p in split_positions:
                        pred_values.append(
                            self.model(
                                imgs, p.unsqueeze(0), matricies, params, num_images
                            )[0]
                            .data.cpu()
                            .numpy()
                        )
                    pred_values = np.concatenate(pred_values)

                    # render prediction
                    on_positions = orig_positions[pred_values > 0.4]
                    on_positions = (
                        on_positions[:, 0],
                        on_positions[:, 1],
                        on_positions[:, 2],
                    )
                    voxels = np.zeros(
                        (128, 128, 128)
                    )  # check occupancy of all voxel locations
                    voxels[on_positions] = 1
                    if voxels.sum() > 0:
                        try:
                            # convert prediction from voxels to mesh
                            verts, faces, normals, values = skimage.measure.marching_cubes(
                                voxels
                            )
                            verts = ((verts / 127.0) * 0.8) - 0.4
                            colour = self.renderer.render_object(
                                verts, faces, add_faces=True
                            )
                            self.writer.add_image(
                                f"{split}/prediction_{num_images}",
                                colour / 255.0,
                                self.epoch,
                                dataformats="HWC",
                            )
                        except:
                            print("not possible to render prediciton")

                # render gt shape
                verts, faces, normals, values = skimage.measure.marching_cubes(
                    gt_voxels
                )
                verts = ((verts / 127.0) * 0.8) - 0.4
                colour = self.renderer.render_object(verts, faces, add_faces=True)
                self.writer.add_image(
                    f"{split}/gt", colour / 255.0, 0, dataformats="HWC"
                )

    # render nice view for evaluation
    def eval_render(self):
        location = f'results/{self.cfg.experiment.experiment_id}/{self.cfg.experiment.id.split("@")[-1]}/'
        print("\n" + "*" * 30)
        print(f"Rendering Images to {location}")
        print("*" * 30)

        os.makedirs(location, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            seed = 0
            # loops across mulitple images
            for i in tqdm(range(0, 10)):

                # get input information
                object_name, _ = self.valid_data.object_names[i]
                imgs, matricies, params = self.valid_data.get_img(object_name, seed)
                gt_voxels, positions, orig_positions = self.valid_data.get_all_voxels(
                    object_name
                )
                render_imgs = (
                    (rearrange(imgs, "b c w h -> w (b h) c") * 255).data.cpu().numpy()
                )
                input_img = scipy.ndimage.zoom(render_imgs, (2, 2, 1), order=3)

                # render grund truth image
                verts, faces, normals, values = skimage.measure.marching_cubes(
                    gt_voxels
                )
                verts = ((verts / 127.0) * 0.8) - 0.4
                gt_colour = self.renderer.render_object(verts, faces, add_faces=True)

                # put inputs on cuda
                imgs = imgs.cuda().unsqueeze(0)
                matricies = matricies.cuda().unsqueeze(0)
                params = params.cuda().unsqueeze(0)
                positions = torch.FloatTensor(positions).cuda()
                split_positions = self.split_values(positions, 50)

                # loop over number of input views
                pred_colours = []
                for num_images in range(1, self.cfg.NBV.budget + 1):

                    # loop over chunks of inputs
                    pred_values = []
                    for p in split_positions:
                        pred_values.append(
                            self.model(
                                imgs, p.unsqueeze(0), matricies, params, num_images
                            )[0]
                            .data.cpu()
                            .numpy()
                        )
                    pred_values = np.concatenate(pred_values)

                    # render prediction
                    on_positions = orig_positions[pred_values > 0.4]
                    on_positions = (
                        on_positions[:, 0],
                        on_positions[:, 1],
                        on_positions[:, 2],
                    )
                    voxels = np.zeros((128, 128, 128))
                    voxels[on_positions] = 1
                    if voxels.sum() > 0:
                        verts, faces, normals, values = skimage.measure.marching_cubes(
                            voxels
                        )  # convert voxels to mesh
                        verts = ((verts / 127.0) * 0.8) - 0.4
                        colour = self.renderer.render_object(
                            verts, faces, smoothing=True, add_faces=True
                        )
                        pred_colours.append(colour)
                    else:
                        pred_colours.append(np.ones((256, 256, 3)))
                pred_colours = (
                    np.stack(pred_colours)
                    .transpose(1, 0, 2, 3)
                    .reshape(256, 5 * 256, 3)
                )
                picture = np.zeros((2 * 256, 6 * 256, 3))
                picture[128 : 128 + 256, :256] = gt_colour
                picture[:256, 256:] = input_img.clip(0, 255)
                picture[256:, 256:] = pred_colours
                picture = picture.astype(np.uint8)
                Image.fromarray(picture).save(
                    location + f"{i}.png"
                )  # save image to nice location

    #  save model instance
    def save(self, best=False, type=""):
        add = "_" + type + "_best.pt" if best else "_resume.pt"
        check_point = self.savedir + add
        torch.save(
            {
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_loss_iou": self.best_loss_iou,
                "best_loss_bce": self.best_loss_bce,
                "last_improvement": self.last_improvement,
            },
            check_point,
        )

    # load model instance
    def load(self, best=False):
        add = "_iou_best.pt" if best else "_resume.pt"
        check_point = self.savedir + add
        check_point = torch.load(check_point)

        self.model.load_state_dict(check_point["model"], strict=False)
        try:
            self.optimizer.load_state_dict(check_point["optimizer"])
        except:
            print("Did not load optimizer")

        self.epoch = check_point["epoch"]
        self.best_loss_iou = check_point["best_loss_iou"]
        self.best_loss_bce = check_point["best_loss_bce"]
        self.last_improvement = check_point["last_improvement"]

    # check if the latest validation beats the previous, and save model if so
    def check_values(self):

        if self.best_loss_bce >= self.current_loss_bce:
            improvement = self.best_loss_bce - self.current_loss_bce
            print(f"Saving with {improvement:.6f} improvement in bce")
            self.best_loss_bce = self.current_loss_bce
            self.last_improvement = 0
            self.save(best=True, type="bce")

        if self.best_loss_iou <= self.current_loss_iou:
            improvement = self.current_loss_iou - self.best_loss_iou
            print(f"Saving with {improvement:.6f} improvement in iou")
            self.best_loss_iou = self.current_loss_iou
            self.last_improvement = 0
            self.save(best=True, type="iou")
        else:
            self.last_improvement += 1
        if self.last_improvement >= self.cfg.experiment.patience:
            print(f"Over {self.cfg.experiment.patience} steps since last imporvement")
            print("Exiting now")
            return True
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/ABC_3D.yml",
        help="Path to (.yml) config file.",
    )

    parser.add_argument(
        "--limit_data",
        action="store_true",
        default=False,
        help="use less data, for debugging.",
    )
    parser.add_argument("--eval", action="store_true", default=False, help="evaluate")

    parser.add_argument(
        "--reset", action="store_true", default=False, help="reset the training"
    )

    configargs = parser.parse_args()
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    cfg.limit_data = configargs.limit_data
    cfg.reset = configargs.reset
    cfg.eval = configargs.eval

    trainer = Engine(cfg)
    trainer()
