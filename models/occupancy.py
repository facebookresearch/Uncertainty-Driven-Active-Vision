# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from models import CNN

import sys

sys.path.insert(0, "../")
from utils import train_utils


class model(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.encoder = CNN.Image_Encoder(options)
        self.positional_encoding = train_utils.positional_encoding
        self.decoder = Decoder(options)
        self.merger = Merger(options)

    # get image feature maps
    def image_to_embedding(self, imgs, num_imgs):
        imgs = imgs[:, :num_imgs]
        imgs = rearrange(imgs, "b n c h w  -> (b n) c h w")
        embedding = self.encoder.get_embedding(imgs)
        return embedding

    # get occupancy from image embedding directly
    def embedding_to_values(self, embedding, points, matricies, params, num_imgs):

        # remove unused image parameters
        matricies = matricies[:, :num_imgs]
        params = params[:, :num_imgs]

        # use image instances in a batch independently
        matricies = rearrange(matricies, "b n c d  -> (b n) c d")
        params = rearrange(params, "b n c  -> (b n) c")
        points = points.unsqueeze(1).repeat(1, num_imgs, 1, 1)
        points = rearrange(points, "b n p d -> (b n) p d")

        # extract image embedding from feature maps
        embedding = self.encoder.embedding_to_values(embedding, points, matricies)

        # compute positional embedding of points and parameters and concatenate
        pos_embedding = self.positional_encoding(points)
        params_embedding = (
            self.positional_encoding(params)
            .unsqueeze(1)
            .repeat(1, pos_embedding.shape[1], 1)
        )
        pos_embedding = torch.cat((pos_embedding, params_embedding), dim=-1)

        # decode image and positional embedding to single feature vector for each image
        occ = self.decoder(pos_embedding, embedding)
        occ = occ.view(-1, num_imgs, occ.shape[-2], occ.shape[-1])

        # merge features across images to single occupancy prediction for each scene
        pred = self.merger(occ)

        return pred

    def forward(self, imgs, points, matricies, params, num_imgs):

        # remove unused image parameters
        imgs = imgs[:, :num_imgs]
        matricies = matricies[:, :num_imgs]
        params = params[:, :num_imgs]

        # use image instances in a batch independently
        imgs = rearrange(imgs, "b n c h w  -> (b n) c h w")
        matricies = rearrange(matricies, "b n c d  -> (b n) c d")
        params = rearrange(params, "b n c  -> (b n) c")
        points = points.unsqueeze(1).repeat(1, num_imgs, 1, 1)
        points = rearrange(points, "b n p d -> (b n) p d")

        # get image embedding
        embedding = self.encoder(imgs, points, matricies)

        # compute positional embedding of points and parameters and concatenate
        pos_embedding = self.positional_encoding(points)
        params_embedding = (
            self.positional_encoding(params)
            .unsqueeze(1)
            .repeat(1, pos_embedding.shape[1], 1)
        )
        pos_embedding = torch.cat((pos_embedding, params_embedding), dim=-1)

        # decode image and positional embedding to single feature vector for each image
        occ = self.decoder(pos_embedding, embedding)
        occ = occ.view(-1, num_imgs, occ.shape[-2], occ.shape[-1])

        # merge features across images to single occupancy prediction for each scene
        pred = self.merger(occ)
        return pred


# takes in image features and positional embeddings and outputs a single unified representation for each view
class Decoder(nn.Module):
    def __init__(self, options):
        super().__init__()

        self.hidden_size = options.model.hidden_size
        self.fc_p = nn.Linear(210, self.hidden_size // 2)
        self.fc_z = nn.Linear(options.model.image_embedding, self.hidden_size // 2)

        self.blocks = []
        for i in range(options.model.decoder_layers):
            self.blocks.append(ResnetBlockFC(self.hidden_size))
        self.blocks = nn.ModuleList(self.blocks)

        self.fc_out = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.actvn = F.relu

    def forward(self, p, z):

        p = self.fc_p(p)
        z = self.fc_z(z)
        net = torch.cat((p, z.expand(p.shape)), dim=-1)

        for block in self.blocks:
            net = block(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)[..., : self.hidden_size]

        return out


# standard resnet block
class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


# merges features from each view into single occupancy prediction
class Merger(nn.Module):
    def __init__(self, options):
        super().__init__()

        hidden_size = options.model.hidden_size
        self.blocks = []
        self.num_layers = options.model.decoder_layers

        # deep set layers
        self.blocks.append(nn.Linear(hidden_size, hidden_size))
        for i in range(options.model.decoder_layers - 1):
            self.blocks.append(nn.Linear(hidden_size * 2, hidden_size))
        self.blocks = nn.ModuleList(self.blocks)

        self.fc_out = nn.Linear(hidden_size, 1)
        self.actvn = F.relu
        self.pool = maxpool
        self.options = options

    def forward(self, occ):

        net = occ.clone()
        # deepset pooling
        for l, block in enumerate(self.blocks):
            net = block(self.actvn(net))
            pooled = self.pool(net, dim=1)
            if l < self.num_layers - 1:
                pooled = pooled.unsqueeze(1).expand(net.size())
                net = torch.cat([net, pooled], dim=-1)
        net = pooled
        out = self.fc_out(self.actvn(net).view(occ.shape[0], occ.shape[2], -1))[..., 0]

        return out
