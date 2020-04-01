import os
import random
from typing import Tuple
import json
from torch import Tensor
from scipy import sparse
from torch import nn
from torchvision import transforms as transforms
import os.path as osp
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageDraw
from datasets import BaseDataset, get_transforms
from datasets.data_utils import (
    get_dir_file_extension,
    remove_top_dir,
    remove_extension,
    find_valid_files,
    decompress_cloth_segment,
    per_channel_transform,
    crop_tensors,
    get_norm_stats,
    to_onehot_tensor,
)

"""
# colour map
label_colours = [(0,0,0)
                # 0=Background
                ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe
"""
class WarpDataset(BaseDataset):
    """ Warp dataset for the warp module of SwapNet """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--input_transforms",
            nargs="+",
            default="none",
            choices=("none", "hflip", "vflip", "affine", "perspective", "all"),
            help="what random transforms to perform on the input ('all' for all transforms)",
        )
        if is_train:
            parser.set_defaults(
                input_transforms=("hflip", "vflip", "affine", "perspective")
            )
        parser.add_argument(
            "--per_channel_transform",
            action="store_true",
            default=True,  # TODO: make this a toggle based on if data is RGB or labels
            help="Perform the transform for each label instead of on the image as a "
            "whole. --cloth_representation must be 'labels'.",
        )
        return parser

    def __init__(self, opt, cloth_dir=None, body_dir=None):
        """

        Args:
            opt:
            cloth_dir: (optional) path to cloth dir, if provided
            body_dir: (optional) path to body dir, if provided
        """
        super().__init__(opt)
        self.fine_width = opt.fine_width
        self.fine_height = opt.fine_height
        self.radius = opt.radius
        # load data list
        im_namess = []
        im_namest = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name1, im_name2, c_name, is_train = line.strip().split()
                # first pair
                if(is_train == "train"):
                    im_namess.append(im_name1)
                    im_namest.append(im_name2)
                    c_names.append(c_name)
                    # second pair
                    im_namest.append(im_name1)
                    im_namess.append(im_name2)
                    c_names.append(c_name)

        self.im_namess = im_namess
        self.im_namest = im_namest
        self.c_names = c_names
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transformmask = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,), (0.5,))])
        """
        self.cloth_dir = cloth_dir if cloth_dir else os.path.join(opt.dataroot, "cloth")
        print("cloth dir", self.cloth_dir)
        extensions = [".npz"] if self.opt.cloth_representation == "labels" else None
        print("Extensions:", extensions)
        self.cloth_files = find_valid_files(self.cloth_dir, extensions)
        if not opt.shuffle_data:
            self.cloth_files.sort()

        self.body_dir = body_dir if body_dir else os.path.join(opt.dataroot, "body")
        if not self.is_train:  # only load these during inference
            self.body_files = find_valid_files(self.body_dir)
            if not opt.shuffle_data:
                self.body_files.sort()
        print("body dir", self.body_dir)
        self.body_norm_stats = get_norm_stats(os.path.dirname(self.body_dir), "body")
        opt.body_norm_stats = self.body_norm_stats
        self._normalize_body = transforms.Normalize(*self.body_norm_stats)
        self.cloth_transform = get_transforms(opt)
        
        """





    def __len__(self):
        """
        Get the length of usable images. Note the length of cloth and body segmentations should be same
        """
        if not self.is_train:
            return min(len(self.c_names), len(self.im_namess))
        else:
            return len(self.im_namess)

    def _load_cloth(self, index) -> Tuple[str, Tensor, Tensor]:
        """
        Loads the cloth file as a tensor
        """
        cloth_file = self.cloth_files[index]
        target_cloth_tensor = decompress_cloth_segment(
            cloth_file, self.opt.cloth_channels
        )
        if self.is_train:
            # during train, we want to do some fancy transforms
            if self.opt.dataset_mode == "image":
                # in image mode, the input cloth is the same as the target cloth
                input_cloth_tensor = target_cloth_tensor.clone()
            elif self.opt.dataset_mode == "video":
                # video mode, can choose a random image
                cloth_file = self.cloth_files[random.randint(0, len(self)) - 1]
                input_cloth_tensor = decompress_cloth_segment(
                    cloth_file, self.opt.cloth_channels
                )
            else:
                raise ValueError(self.opt.dataset_mode)

            # apply the transformation for input cloth segmentation
            if self.cloth_transform:
                input_cloth_tensor = self._perform_cloth_transform(input_cloth_tensor)

            return cloth_file, input_cloth_tensor, target_cloth_tensor
        else:
            # during inference, we just want to load the current cloth
            return cloth_file, target_cloth_tensor, target_cloth_tensor

    def _load_body(self, index):
        """ Loads the body file as a tensor """
        if self.is_train:
            # use corresponding strategy during train
            cloth_file = self.cloth_files[index]
            body_file = get_corresponding_file(cloth_file, self.body_dir)
        else:
            # else we have to load by index
            body_file = self.body_files[index]
        as_pil_image = Image.open(body_file).convert("RGB")
        body_tensor = self._normalize_body(transforms.ToTensor()(as_pil_image))
        return body_file, body_tensor

    def _perform_cloth_transform(self, cloth_tensor):
        """ Either does per-channel transform or whole-image transform """
        if self.opt.per_channel_transform:
            return per_channel_transform(cloth_tensor, self.cloth_transform)
        else:
            raise NotImplementedError("Sorry, per_channel_transform must be true")
            # return self.input_transform(cloth_tensor)

    def __getitem__(self, index):
        """
        :returns:
            For training, return (input) AUGMENTED cloth seg, (input) body seg and (target) cloth seg
            of the SAME image
            For inference (e.g validation), return (input) cloth seg and (input) body seg
            of 2 different images
        """
        c_name = self.c_names[index]
        im_names = self.im_namess[index]
        im_namet = self.im_namest[index]

        parse_names = im_names.replace("jpg","png")
        parse_namet = im_namet.replace("jpg","png")

        im_parset = Image.open(osp.join(self.opt.dataroot, 'all_parsing', parse_namet))
        parse_arrayt = np.array(im_parset)
        #print("Max value:", np.amax(parse_arrayt))
        #print("Min value:", np.amin(parse_arrayt))
        parse_arrayt = sparse.csr_matrix(parse_arrayt)
        #print(parse_arrayt.shape)
        parse_arrayt = to_onehot_tensor(parse_arrayt, 20)
        #parse_arrayt = self.transformmask(parse_arrayt)  # [0,1]

        im_parse = Image.open(osp.join(self.opt.dataroot, 'all_parsing', parse_names))
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_hair = (parse_array == 2).astype(np.float32)
        parse_face = (parse_array == 13).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                     (parse_array == 2).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)

        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transformmask(parse_shape)  # [-1,1]
        pface = self.transformmask(parse_face)  # [0,1]

        phair = self.transformmask(parse_hair)  # [0,1]

        c = Image.open(osp.join(self.opt.dataroot, 'all', c_name))
        c = self.transform(c)  # [-1,1]

        # load pose points
        pose_name = im_namet.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.opt.dataroot, 'all_person_clothes_keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transformmask(one_map)
            pose_map[i] = one_map[0]



        # cloth-agnostic representation
        agnostic = torch.cat([phair,pface,shape, c, pose_map], 0) # 1 1 1 3 20


        return {
            "cloth": c,  # for input
            "phair": phair,
            "pface": pface,
            "shape": shape,
            "pose_map": pose_map,
            "agnostic": agnostic,
            "parse_arrayt": parse_arrayt
        }


def get_corresponding_file(original, target_dir, target_ext=None):
    """
    Say an original file is
        dataroot/subject/body/SAMPLE_ID.jpg

    And we want the corresponding file
        dataroot/subject/cloth/SAMPLE_ID.npz

    The corresponding file is in target_dir dataroot/subject/cloth, so we replace the
    top level directories with the target dir

    Args:
        original:
        target_dir:
        target_ext:

    Returns:

    """
    # number of top dir to replace
    num_top_parts = len(target_dir.split(os.path.sep))
    # replace the top dirs
    top_removed = remove_top_dir(original, num_top_parts)
    target_file = os.path.join(target_dir, top_removed)
    # extension of files in the target dir
    if not target_ext:
        target_ext = get_dir_file_extension(target_dir)
    # change the extension
    target_file = remove_extension(target_file) + target_ext
    return target_file
