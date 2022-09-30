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
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image


import sys
sys.path.insert(0, "../")
from utils import CfgNode, train_utils, rays
from models import occupancy_and_colour


class Engine(Checkpointable):
    def __init__(self, cfg):
        self.cfg = cfg

        # initial settings
        self.best_loss = 1
        self.epoch = 0
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

        # get datasert
        self.get_loaders()

        # make model
        self.model = occupancy_and_colour.model(self.cfg)
        self.model.cuda()

        # optimizer
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params, lr=self.cfg.optimizer.lr, weight_decay=0)

        # evaluate model
        if self.cfg.eval:
            self.load(best=True)
            self.validate()
            self.eval_render()
            exit()

        # restart training
        if not cfg.reset:
            self.resume()

        # training loop
        start_iter = self.epoch
        for i in range(start_iter, cfg.experiment.num_epochs):
            self.train()
            if i % 2 == 0 and i > start_iter:
                self.validate()
                if self.check_values():
                    return
                self.render()
                self.save(best=False)
            self.epoch += 1

    def get_loaders(self):

        # training dataloader
        train_data = data_loader.data(self.cfg, set_type="train")
        train_data[0]
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
        average_loss = []

        for k, batch in enumerate(tqdm(self.train_loader, smoothing=0)):
            self.optimizer.zero_grad()

            # inputs
            input_imgs = batch["imgs"][:, :-1].cuda()
            input_matricies = batch["matricies"][:, :-1].cuda()
            input_params = batch["params"][:, :-1].cuda()
            num_images = np.random.choice(np.arange(1, self.cfg.NBV.budget + 1))
            output_imgs = batch["imgs"][:, -1].cuda()
            output_params = batch["params"][:, -1].cuda()

            # get ray values for volume rendering and sumsample for training
            ray_points, ray_masks, _ = rays.get_rays(self, output_params[:, :3])
            gt_values, ray_points, ray_masks = rays.subsample_rays(
                self, ray_points, ray_masks, output_imgs, self.cfg.loss.num_rays
            )

            # inference
            pred_values = self.model(
                input_imgs,
                ray_points,
                input_matricies,
                input_params,
                output_params,
                num_images,
            )
            pred_colour = train_utils.nerf_rendering(pred_values, ray_masks)

            # loss
            nerf_loss = ((gt_values - pred_colour) ** 2).mean()
            loss = nerf_loss

            # optimize
            loss.backward()
            self.optimizer.step()

            # log
            average_loss.append(loss.item())
            if k % 10 == 0 or self.cfg.limit_data:
                message = f"Train || Epoch: {self.epoch}, mse: {nerf_loss.item():.5f} "
                message += f"  || best_loss:  {self.best_loss :.5f}"
                tqdm.write(message)
        average_loss = np.array(average_loss).mean()
        self.writer.add_scalar("train/loss", average_loss, self.epoch)

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            loss_total = [0 for _ in range(self.cfg.NBV.budget)]
            iou_total = [0 for _ in range(self.cfg.NBV.budget)]

            for k, batch in enumerate(tqdm(self.valid_loader, smoothing=0)):
                # inputs
                input_imgs = batch["imgs"][:, :-1].cuda()
                input_matricies = batch["matricies"][:, :-1].cuda()
                input_params = batch["params"][:, :-1].cuda()
                output_imgs = batch["imgs"][:, -1].cuda()
                output_params = batch["params"][:, -1].cuda()
                positions = batch["positions"].cuda()
                gt_values_iou = batch["gt_values"].cuda()

                # get ray values for volume rendering and sumsample for training
                all_points, all_masks, _ = rays.get_rays(self, output_params[:, :3])
                gt_values, ray_points, ray_masks = rays.subsample_rays(
                    self,
                    all_points,
                    all_masks,
                    output_imgs,
                    self.cfg.loss.num_rays,
                    seed=k,
                )

                # loop over number of view
                for j in range(1, self.cfg.NBV.budget + 1):
                    # inference
                    pred_values = self.model(
                        input_imgs,
                        ray_points,
                        input_matricies,
                        input_params,
                        output_params,
                        j,
                        train=False,
                    )
                    pred_colour = train_utils.nerf_rendering(pred_values, ray_masks)

                    loss = ((gt_values - pred_colour) ** 2).mean()
                    loss_total[j - 1] += loss.item()

                    pred_values = []
                    # loop over chunks
                    for i in range(10):
                        pred_values.append(
                            self.model(
                                input_imgs,
                                positions[:, i * 10000 : i * 10000 + 10000],
                                input_matricies,
                                input_params,
                                output_params,
                                j,
                                train=False,
                            )
                        )
                    values = torch.cat(pred_values, dim=1)
                    values = torch.sigmoid(values[..., 0])
                    loss_iou = train_utils.get_iou(
                        values, gt_values_iou, threshold=0.4, apply_threshold=True
                    ).mean()
                    iou_total[j - 1] += loss_iou.item()

            # logs
            loss_total = np.array(loss_total) / float(k + 1)
            iou_total = np.array(iou_total) / float(k + 1)
            message = f"Validation Total || Epoch: {self.epoch}, mse: {loss_total}, iou: {iou_total}"
            message += f" || best loss:  {self.best_loss :.5f}"
            tqdm.write(message)

            if not np.isinf(np.array(loss_total).mean()):
                for j in range(self.cfg.NBV.budget):
                    self.writer.add_scalar(
                        f"validation/loss_{j}", loss_total[j], self.epoch
                    )

                cur_loss = np.array(loss_total).mean()
                self.writer.add_scalar(f"validation/loss_mean", cur_loss, self.epoch)
                self.current_loss = cur_loss
            else:
                print("the value is inf so we are skipping for now :)")
                self.current_loss = self.current_loss - 0.1

    # split values into chunks
    def split_values(self, inputs, splits):
        values = []
        inputs = inputs.view(-1, 3)
        length = inputs.shape[0] // splits
        for i in range(splits):
            values.append(inputs[i * length : i * length + length])
        values.append(inputs[(splits) * length :])

        total = 0
        for v in values:
            total += v.shape[0]
        assert total == inputs.shape[0]
        return values

    # render a view for logging
    def render(self):
        print("\n" + "*" * 30)
        print(f"Rendering Images")
        print("*" * 30)
        self.model.eval()
        with torch.no_grad():
            seed = 0
            # rendering object frm training and validation set
            for e, loader in enumerate([self.train_data, self.valid_data]):
                split = "train" if e == 0 else "valid"

                # inputs
                object_name, _ = loader.object_names[0]
                imgs, matricies, params = loader.get_img(object_name, seed)
                input_imgs = imgs[:-1].cuda().unsqueeze(0)
                input_params = params[:-1].cuda().unsqueeze(0)
                input_matricies = matricies[:-1].cuda().unsqueeze(0)
                output_img = imgs[-1].cuda()
                output_params = params[-1].cuda().unsqueeze(0)

                # save input and target views
                render_imgs = rearrange(imgs, "b c w h -> c w (b h)")
                self.writer.add_image(
                    f"{split}/input", render_imgs.data.cpu().numpy(), 0
                )
                self.writer.add_image(f"{split}/target", imgs[-1].data.cpu().numpy(), 0)

                # get ray values for vlume rendering
                ray_points, ray_masks, _ = rays.get_rays(self, output_params[:, :3])
                split_positions = self.split_values(ray_points, 50)
                for num_images in range(1, self.cfg.NBV.budget + 1):
                    # get values from chunks
                    pred_values = []
                    for p in split_positions:
                        values = self.model(
                            input_imgs,
                            p.unsqueeze(0),
                            input_matricies,
                            input_params,
                            output_params,
                            num_images,
                        )[0]
                        pred_values.append(values)
                    pred_values = torch.cat(pred_values).view(
                        list(ray_masks.shape) + [4]
                    )
                    # render views
                    colour = train_utils.nerf_rendering(pred_values, ray_masks).view(
                        128, 128, 3
                    )
                    self.writer.add_image(
                        f"{split}/prediction_{num_images}",
                        colour,
                        self.epoch,
                        dataformats="HWC",
                    )

    # render view for evaluation
    def eval_render(self):
        self.model.eval()
        location = f'results/{str(self.cfg.experiment.id).split("@")[-1]}/'
        if not os.path.exists(location):
            os.makedirs(location)
        print("\n" + "*" * 30)
        print(f"Rendering Images at {location}")
        print("*" * 30)

        with torch.no_grad():
            seed = 0
            # loop over objects
            for i in tqdm(range(10)):
                object_name, _ = self.valid_data.object_names[i]
                imgs, matricies, params = self.valid_data.get_img(object_name, seed)
                render_imgs = (
                    (rearrange(imgs[:-1], "b c w h -> w (b h) c") * 255)
                    .data.cpu()
                    .numpy()
                )
                gt_colour = (
                    (rearrange(imgs[-1:], "b c w h -> w (b h) c") * 255)
                    .data.cpu()
                    .numpy()
                )

                # inputs
                input_imgs = imgs[:-1].cuda().unsqueeze(0)
                input_params = params[:-1].cuda().unsqueeze(0)
                input_matricies = matricies[:-1].cuda().unsqueeze(0)
                output_params = params[-1].cuda().unsqueeze(0)
                ray_points, ray_masks, _ = rays.get_rays(self, output_params[:, :3])
                split_positions = self.split_values(ray_points, 20)

                preds = []
                for num_images in range(1, self.cfg.NBV.budget + 1):
                    # get image embedding
                    embedding = self.model.image_to_embedding(input_imgs, num_images)
                    # define function from embedding to colour and coocupancy values
                    fun = lambda x: self.model.embedding_to_values(
                        embedding,
                        x,
                        input_matricies,
                        input_params,
                        output_params,
                        num_images,
                    )[0]

                    # predict over chunks
                    pred_values = []
                    for p in split_positions:
                        values = fun(p.unsqueeze(0))
                        pred_values.append(values)
                    pred_values = torch.cat(pred_values).view(
                        list(ray_masks.shape) + [4]
                    )
                    # render views
                    colour = train_utils.nerf_rendering(pred_values, ray_masks).view(
                        128, 128, 3
                    )
                    preds.append(colour)

                colour = torch.cat(preds, dim=1)
                colour = colour.data.cpu().numpy()
                pred_colours = (colour * 255).astype(np.uint8)

                # save image of predictions
                picture = np.zeros((2 * 128, 6 * 128, 3))
                picture[64 : 64 + 128, :128] = gt_colour
                picture[:128, 128:] = render_imgs.clip(0, 255)
                picture[128:, 128:] = pred_colours
                picture = picture.astype(np.uint8)
                Image.fromarray(picture).save(location + f"{i}.png")

    # save model
    def save(self, best=False, type=""):
        add = "_" + type + "_best.pt" if best else "_resume.pt"
        check_point = self.savedir + add
        torch.save(
            {
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "last_improvement": self.last_improvement,
            },
            check_point,
        )

    # load model
    def load(self, best=False):
        add = "_loss_best.pt" if best else "_resume.pt"
        check_point = self.savedir + add
        check_point = torch.load(check_point)

        self.model.load_state_dict(check_point["model"], strict=False)
        try:
            self.optimizer.load_state_dict(check_point["optimizer"])
        except:
            print("Did not load optimizer")

        self.epoch = check_point["epoch"]
        self.best_loss = check_point["best_loss"]
        self.last_improvement = check_point["last_improvement"]

    # check if the latest validation beats the previous, and save model if so
    def check_values(self):
        if self.best_loss >= self.current_loss:
            improvement = -(self.current_loss - self.best_loss)
            print(f"Saving with {improvement:.6f} improvement")
            self.best_loss = self.current_loss
            self.last_improvement = 0
            self.save(best=True, type="loss")
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
        default="../configs/ABC_2D.yml",
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
        "--reset", action="store_true", default=False, help="reset training"
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
