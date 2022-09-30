# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrender
import trimesh
import math
from PIL import Image


class Renderer:
    def __init__(self, cameraResolution=[256, 256]):
        self.scene = self.init_scene()
        self.object_nodes = []
        self.init_camera()
        self.r = pyrender.OffscreenRenderer(cameraResolution[0], cameraResolution[1])

    # scene is initialized with fixed lights, this can be easily changed to match the desired environment
    def init_scene(self):
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
        light_pose = self.euler2matrix(
            angles=[0, 0, 0], translation=[0, -0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=2)
        scene.add(light, pose=light_pose)

        light_pose = self.euler2matrix(
            angles=[0, 0, 0], translation=[0, 0.8, 0.3], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=2)
        scene.add(light, pose=light_pose)

        light_pose = self.euler2matrix(
            angles=[0, 0, 0], translation=[-1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=2)
        scene.add(light, pose=light_pose)

        light_pose = self.euler2matrix(
            angles=[0, 0, 0], translation=[1, 0, 1], xyz="xyz", degrees=False
        )
        light = pyrender.PointLight(color=np.ones(3), intensity=2)
        scene.add(light, pose=light_pose)

        return scene

    # converts a euler rotation to a rotation matrix
    def euler2matrix(
        self, angles=[0, 0, 0], translation=[0, 0, 0], xyz="xyz", degrees=False
    ):
        r = R.from_euler(xyz, angles, degrees=degrees)
        pose = np.eye(4)
        pose[:3, 3] = translation
        pose[:3, :3] = r.as_matrix()
        return pose

    # initializes the camera parameters
    def init_camera(self):
        camera = pyrender.PerspectiveCamera(
            yfov=60.0 / 180.0 * np.pi, znear=0.01, zfar=10.0, aspectRatio=1.0
        )
        camera_pose = self.euler2matrix(
            xyz="xyz", angles=[0, 0, 0], translation=[0, 0, 1], degrees=True
        )

        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(camera_node)
        self.scene.main_camera_node = camera_node
        self.camera = camera_node
        initial_matrix = R.from_euler("xyz", [45.0, 0, 180.0], degrees=True).as_matrix()
        self.update_camera_pose([0, 0.6, 0.6], initial_matrix)

    # use all face vertex rotations to name normals always face outwards
    def add_faces(self, mesh):
        verts, faces = mesh.vertices, mesh.faces
        f1 = np.array(faces[:, 0]).reshape(-1, 1)
        f2 = np.array(faces[:, 1]).reshape(-1, 1)
        f3 = np.array(faces[:, 2]).reshape(-1, 1)
        faces_2 = np.concatenate((f1, f3, f2), axis=-1)
        faces_3 = np.concatenate((f3, f2, f1), axis=-1)
        faces = np.concatenate((faces, faces_2, faces_3), axis=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        return mesh

    # get random location in search space
    def random_position(self, radius=1, num=1, seed=None):
        if num == 1:
            if seed is not None:
                vec = np.random.RandomState(seed).rand(3) * 2 - 1
            else:
                vec = np.random.rand(3) * 2 - 1
            # vec[-1] = abs(vec[-1])
            vec /= np.linalg.norm(vec, axis=0)
        else:
            if seed is not None:
                vec = np.random.RandomState(seed).rand(num, 3) * 2 - 1
            else:
                vec = np.random.rand(num, 3) * 2 - 1

            vec /= np.linalg.norm(vec, axis=-1).reshape(vec.shape[0], 1)
        vec *= radius
        return vec

    # add object to the scene
    def add_object(
        self,
        mesh,
        position=[0, 0, 0],
        orientation=[0, 0, 0],
        colour=[228, 217, 111, 255],
        add_faces=False,
    ):
        if add_faces:
            mesh = self.add_faces(mesh)
        mesh.visual.vertex_colors = colour
        mesh = pyrender.Mesh.from_trimesh(mesh)
        pose = self.euler2matrix(angles=orientation, translation=position)
        obj_node = pyrender.Node(mesh=mesh, matrix=pose)
        self.scene.add_node(obj_node)
        self.object_nodes.append(obj_node)

    # remove object from the scene
    def remove_objects(self):
        for obj in self.object_nodes:
            self.scene.remove_node(obj)
        self.object_nodes = []

    # get camera rotation to point to center of object given position
    def cam_from_positions(self, new_cam_positions):
        camera_direction = np.array(new_cam_positions)
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        camera_right = np.cross(np.array([0.0, 0.0, 1.0]), camera_direction)
        camera_right = camera_right / np.linalg.norm(camera_right)
        camera_up = np.cross(camera_direction, camera_right)
        camera_up = camera_up / np.linalg.norm(camera_up)
        rotation_transform = np.zeros((4, 4))
        rotation_transform[0, :3] = camera_right
        rotation_transform[1, :3] = camera_up
        rotation_transform[2, :3] = camera_direction
        rotation_transform[-1, -1] = 1
        translation_transform = np.eye(4)
        translation_transform[:3, -1] = -new_cam_positions
        l = np.matmul(rotation_transform, translation_transform)
        rot = R.from_matrix(l[:3, :3].transpose())
        rot = rot.as_euler("xyz", degrees=True)
        return rot

    # update the camera
    def update_camera_pose(self, position, orientation):
        pose = np.eye(4)
        if np.array(orientation).shape == (3,):
            orientation = R.from_euler("xyz", orientation, degrees=True).as_matrix()
        pose[:3, 3] = position
        pose[:3, :3] = orientation
        self.camera.matrix = pose

    # render image of the object
    def render_object(
        self, verts, faces, camera_params=None, smoothing=False, add_faces=False
    ):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if smoothing:
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=15)
        self.add_object(mesh, add_faces=add_faces)
        colour = self.render()
        self.remove_objects()
        return colour

    # render the scene
    def render(self, get_depth=False):
        colour, depth = self.r.render(self.scene)
        if get_depth:
            return colour, depth
        return colour
