# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import numpy as np

# basic CNN layer template
def CNN_layer(f_in, f_out, k, stride=1, simple=False, padding=1):
    layers = []
    if not simple:
        layers.append(nn.BatchNorm2d(int(f_in)))
        layers.append(nn.ReLU(inplace=True))
    layers.append(
        nn.Conv2d(int(f_in), int(f_out), kernel_size=k, padding=padding, stride=stride)
    )
    return nn.Sequential(*layers)


# network for making image features for vertex feature vectors
class Image_Encoder(nn.Module):
    def __init__(self, args):
        super(Image_Encoder, self).__init__()

        # CNN sizes
        cur_size = 3
        next_size = args.model.initial_size
        args.model.CNN_ker_size = args.model.CNN_ker_size

        # layers for the CNN
        layers = []
        layers.append(
            CNN_layer(
                cur_size, cur_size, args.model.CNN_ker_size, stride=1, simple=True
            )
        )
        for i in range(args.model.num_CNN_blocks):
            layers.append(
                CNN_layer(cur_size, next_size, args.model.CNN_ker_size, stride=2)
            )
            cur_size = next_size
            next_size = next_size * 2
            for j in range(args.model.layers_per_block - 1):
                layers.append(CNN_layer(cur_size, cur_size, args.model.CNN_ker_size))

        self.args = args
        self.layers = nn.ModuleList(layers)

        x = torch.ones(1, 3, 128, 128)
        features_size = 0
        # layers to select image features from
        layer_selections = [
            len(self.layers) - 1 - (i + 1) * self.args.model.layers_per_block
            for i in range(3)
        ]
        for e, layer in enumerate(self.layers):
            if x.shape[-1] < self.args.model.CNN_ker_size:
                break
            x = layer(x)
            if e in layer_selections:
                features_size += x.shape[1]
        features_size += x.shape[1]
        self.fc = nn.Linear(features_size, args.model.image_embedding)

    # defines image features over vertices from vertex positions, and feature mpas from vision
    def pooling(self, blocks, verts_pos, matricies):
        # convert vertex positions to x,y coordinates in the image, scaled to fractions of image dimension
        ext_verts_pos = torch.cat(
            (
                verts_pos,
                torch.FloatTensor(
                    np.ones([verts_pos.shape[0], verts_pos.shape[1], 1])
                ).cuda(),
            ),
            dim=-1,
        )
        ext_verts_pos = torch.bmm(ext_verts_pos, matricies.permute(0, 2, 1))
        ext_verts_pos[:, :, 2][ext_verts_pos[:, :, 2] == 0] = 0.1
        xs = ext_verts_pos[:, :, 1] / ext_verts_pos[:, :, 2] / 128.0
        xs[torch.isinf(xs)] = 0.5
        ys = ext_verts_pos[:, :, 0] / ext_verts_pos[:, :, 2] / 128.0
        ys[torch.isinf(ys)] = 0.5

        full_features = None
        xs = xs.unsqueeze(2).unsqueeze(3) * 0.5
        ys = ys.unsqueeze(2).unsqueeze(3) * 0.5
        grid = torch.cat([ys, xs], 3)
        grid = grid * 2 - 1

        # extract image features based on vertex projected positions
        for block in blocks:
            features = torch.nn.functional.grid_sample(block, grid, align_corners=True)
            if full_features is None:
                full_features = features
            else:
                full_features = torch.cat((full_features, features), dim=1)
        vert_image_features = full_features[:, :, :, 0].permute(0, 2, 1)
        return vert_image_features

    def get_embedding(self, img):
        x = img
        features = []
        # layers to select image features from
        layer_selections = [
            len(self.layers) - 1 - (i + 1) * self.args.model.layers_per_block
            for i in range(3)
        ]
        for e, layer in enumerate(self.layers):
            # if too many layers are applied the map size will be lower then then kernel size
            if x.shape[-1] < self.args.model.CNN_ker_size:
                break
            x = layer(x)
            # collect feature maps
            if e in layer_selections:
                features.append(x)
        features.append(x)
        return features

    def embedding_to_values(self, embedding, points, matricies):
        features = self.pooling(embedding, points, matricies)
        features = self.fc(features)
        return features

    def forward(self, img, points, matricies):
        x = img
        features = []
        # layers to select image features from
        layer_selections = [
            len(self.layers) - 1 - (i + 1) * self.args.model.layers_per_block
            for i in range(3)
        ]
        for e, layer in enumerate(self.layers):
            # if too many layers are applied the map size will be lower then then kernel size
            if x.shape[-1] < self.args.model.CNN_ker_size:
                break
            x = layer(x)
            # collect feature maps
            if e in layer_selections:
                features.append(x)
        features.append(x)
        features = self.pooling(features, points, matricies)
        features = self.fc(features)

        return features
