from __future__ import annotations

import os
import json
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
    

class EgoGenDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        edit_path: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        aspect_ratio: float = 1.0,
        flip_prob: float = 0.0,
        text_prompt_key: str = 'action',
        additional_cross_attn: bool = False,
        additional_cond_path = None,
    ):
        assert split in ("train", "val", "test")
        self.data_path = data_path
        self.edit_path = edit_path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.aspect_ratio = aspect_ratio
        self.flip_prob = flip_prob
        self.split = split
        
        self.text_prompt_key = text_prompt_key
        self.additional_cross_attn = additional_cross_attn
        self.additional_cond_path = additional_cond_path
        self.max_text_length = 77

        with open(self.edit_path, 'r') as infile:
            self.edits = json.load(infile)

        self.actions = [(k1, k2) for k1 in self.edits for k2 in self.edits[k1]]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, i: int) -> dict[str, Any]:
        clip_id, action_id = self.actions[i]

        prompt = self.edits[clip_id][action_id][self.text_prompt_key]

        image_0_path = os.path.join(self.data_path, self.edits[clip_id][action_id]['image_0'])
        image_1_path = os.path.join(self.data_path, self.edits[clip_id][action_id]['image_1'])

        image_0 = Image.open(image_0_path)
        image_1 = Image.open(image_1_path)
        w, h =image_0.size

        height = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        width = round(height * self.aspect_ratio)
        image_0 = image_0.resize((width, height), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((width, height), Image.Resampling.LANCZOS)
        
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop((self.crop_res, round(self.aspect_ratio * self.crop_res)))
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        out = dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))
        if self.additional_cross_attn:
            out['edit']['c_add_crossattn'] = dict()
            for k in sorted(self.additional_cond_path.keys()):
                add_cond_path = os.path.join(self.additional_cond_path[k], self.edits[clip_id][action_id]['image_0'].replace('jpg', 'npy'))
                add_cond = np.load(add_cond_path)

                if k == 'llava_text':
                    if add_cond.ndim == 3:
                        add_cond = add_cond[:, -1, :]

                    if add_cond.shape[0] > self.max_text_length:
                        add_cond = add_cond[:self.max_text_length, :]
                    elif add_cond.shape[0] < self.max_text_length:
                        add_cond_pad = np.zeros(shape=(self.max_text_length-add_cond.shape[0], add_cond.shape[1]), dtype=add_cond.dtype)
                        add_cond = np.concatenate([add_cond, add_cond_pad], axis=0)

                out['edit']['c_add_crossattn'][k] = torch.tensor(add_cond).float()

        return out


class EgoGenDatasetEval(Dataset):
    def __init__(
        self,
        data_path: str,
        edit_path: str,
        split: str = "test",
        res: int = 256,
        aspect_ratio: float = 1.0,
        text_prompt_key: str = 'action',
        additional_cross_attn: bool = False,
        additional_cond_path = None,
    ):
        assert split in ("train", "val", "test")
        self.data_path = data_path
        self.edit_path = edit_path
        self.res = res
        self.aspect_ratio = aspect_ratio

        self.text_prompt_key = text_prompt_key
        self.additional_cross_attn = additional_cross_attn
        self.additional_cond_path = additional_cond_path
        self.max_text_length = 77

        with open(self.edit_path, 'r') as infile:
            self.edits = json.load(infile)

        self.actions = [(k1, k2) for k1 in self.edits for k2 in self.edits[k1]]

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, i: int) -> dict[str, Any]:
        clip_id, action_id = self.actions[i]

        prompt = self.edits[clip_id][action_id][self.text_prompt_key]

        image_0_path = os.path.join(self.data_path, self.edits[clip_id][action_id]['image_0'])
        image_1_path = os.path.join(self.data_path, self.edits[clip_id][action_id]['image_1'])
        
        image_0 = Image.open(image_0_path)
        w, h =image_0.size

        height = torch.randint(self.res, self.res + 1, ()).item()
        width = round(height * self.aspect_ratio)
        image_0 = image_0.resize((width, height), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        out = dict(image_0=image_0, edit=prompt, image_0_path=image_0_path, image_1_path=image_1_path)
        if self.additional_cross_attn:
            out['add_cross_attn'] = dict()
            for k in sorted(self.additional_cond_path.keys()):
                add_cond_path = os.path.join(self.additional_cond_path[k], self.edits[clip_id][action_id]['image_0'].replace('jpg', 'npy'))
                add_cond = np.load(add_cond_path)

                if k == 'llava_text':
                    if add_cond.ndim == 3:
                        add_cond = add_cond[:, -1, :]

                    if add_cond.shape[0] > self.max_text_length:
                        add_cond = add_cond[:self.max_text_length, :]
                    elif add_cond.shape[0] < self.max_text_length:
                        add_cond_pad = np.zeros(shape=(self.max_text_length-add_cond.shape[0], add_cond.shape[1]), dtype=add_cond.dtype)
                        add_cond = np.concatenate([add_cond, add_cond_pad], axis=0)

                out['add_cross_attn'][k] = torch.tensor(add_cond).float()

        return out
