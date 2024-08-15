from __future__ import annotations

import json
import sys
import os
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything

sys.path.append("./")
sys.path.append("./stable_diffusion")

from ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v for k, v in sd.items()}
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print("missing keys:", m)
        print("unexpected keys:", u)
    return model


class ImageEditor(nn.Module):
    def __init__(self, config, ckpt, vae_ckpt=None):
        super().__init__()
        
        self.model = load_model_from_config(config, ckpt, vae_ckpt, verbose=True)
        self.model.eval().cuda()
        self.model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(self.model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

    def forward(
        self,
        image: torch.Tensor,
        edit: str,
        additional_cross_attn = None,
        scale_txt: float = 7.5,
        scale_img: float = 1.0,
        steps: int = 100,
    ) -> torch.Tensor:
        assert image.dim() == 3
        assert image.size(1) % 64 == 0
        assert image.size(2) % 64 == 0
        with torch.no_grad(), autocast("cuda"), self.model.ema_scope():
            cond = {
                "c_crossattn": [self.model.get_learned_conditioning([edit])],
                "c_concat": [self.model.encode_first_stage(image[None]).mode()],
            }
            
            uncond = {
                "c_crossattn": [self.model.get_learned_conditioning([""])],
                "c_concat": [torch.zeros_like(cond["c_concat"][0])],
            }

            if additional_cross_attn is not None:
                if len(additional_cross_attn.keys()) == 2 and 'llava_image' in additional_cross_attn.keys() and 'llava_text' in additional_cross_attn.keys():
                    additional_cross_attn_img = self.model.projection_img(additional_cross_attn['llava_image'][None])  # get float16
                    additional_cross_attn_txt = self.model.projection_txt(additional_cross_attn['llava_text'][None])  # get float16
                    additional_cross_attn_txt = self.model.self_attn_mix(additional_cross_attn_txt)
                    cond['c_crossattn'] = [torch.cat(cond['c_crossattn'] + [additional_cross_attn_img.float(), additional_cross_attn_txt.float()], dim=1)]

                    uncond_add_cross_attn_img = torch.zeros_like(additional_cross_attn_img)
                    uncond_add_cross_attn_txt = torch.zeros_like(additional_cross_attn_txt)
                    uncond['c_crossattn'] = [torch.cat(uncond['c_crossattn'] + [uncond_add_cross_attn_img.float(), uncond_add_cross_attn_txt.float()], dim=1)]

                else:
                    raise NotImplementedError

            extra_args = {
                "uncond": uncond,
                "cond": cond,
                "image_cfg_scale": scale_img,
                "text_cfg_scale": scale_txt,
            }
            sigmas = self.model_wrap.get_sigmas(steps)
            x = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            x = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, x, sigmas, extra_args=extra_args)
            x = self.model.decode_first_stage(x)[0]
            return x


def inference(config, model_path,  vae_ckpt, n_chunk = 1, chunk_idx = 1):
    seed, steps = config.inference.seed, config.inference.steps
    seed_everything(seed)
    print(f'seed={seed}')
    
    editor = ImageEditor(config, model_path, vae_ckpt).cuda()
    with open(config.data.metadata_path, 'r') as infile:
        metadata = json.load(infile)

    for scale_txt in config.inference.scales_txt:
        for scale_img in config.inference.scales_img:
            dataset = instantiate_from_config(config.data)
            print(f'Additional conditioning: {dataset.additional_cond_path}, text_prompt_key: {dataset.text_prompt_key}')
            print(f'Processing t={scale_txt}, i={scale_img}')

            perm = list(range((chunk_idx-1) * len(dataset) // n_chunk, chunk_idx * len(dataset) // n_chunk))
            count = 0
            i = 0

            pbar = tqdm(total=len(perm))
            while count < len(perm):
                idx = perm[i]
                sample = dataset[idx]
                sample['image_0'] = sample['image_0'].cuda()
                if 'add_cross_attn' in sample.keys():
                    if isinstance(sample['add_cross_attn'], torch.Tensor):
                        sample['add_cross_attn'] = sample['add_cross_attn'].cuda()
                    elif isinstance(sample['add_cross_attn'], dict):
                        for k in sample['add_cross_attn']:
                            sample['add_cross_attn'][k] = sample['add_cross_attn'][k].cuda()
                    else:
                        raise NotImplementedError

                i += 1
                count += 1
                
                if 'add_cross_attn' in sample:
                    gen = editor(sample["image_0"], sample["edit"], sample['add_cross_attn'], scale_txt=scale_txt, scale_img=scale_img, steps=steps)  # use text and feature conditioning
                else:
                    gen = editor(sample["image_0"], sample["edit"], scale_txt=scale_txt, scale_img=scale_img, steps=steps)  # use text conditioning only

                clip_id = str(sample["image_1_path"]).split('/')[-3]
                H, W = metadata[clip_id]['height'], metadata[clip_id]['width']
                res = config.data.params.res
                h, w = res, round(W / H * res)
                gen = F.interpolate(gen.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False)
                gen = gen.squeeze(0)

                # save generated images
                gen_norm = (gen - gen.min()) / (gen.max() - gen.min()) * 255
                gen_norm = gen_norm.detach().cpu().numpy()
                gen_norm = np.transpose(gen_norm, axes=[1, 2, 0])
                im = Image.fromarray(gen_norm.astype(np.uint8))
                # save_path = os.path.join(config.inference.output_path, model_path.split('/')[-4], 'analysis', model_path.split('/')[-1][:-5]+f'-e={seed}-s={steps}-si={scale_img}', 'images', *sample["image_1_path"].split('/')[-3:])
                save_path = os.path.join(model_path.replace('.ckpt', ''), model_path.split('/')[-1][:-5]+f'-e={seed}-s={steps}-si={scale_img}', 'images', *sample["image_1_path"].split('/')[-3:])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                im.save(save_path)
                pbar.update()
            pbar.close()

    return


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Configuration file, e.g. configs/generate_ego4d.yaml")
    parser.add_argument("--ckpt", type=str, help="Model checkpoint, e.g. logs/train_ego4d_base/checkpoints/trainstep_checkpoints/epoch=000119-step=000009999.ckpt")
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--n_chunk", default=1, type=int)
    parser.add_argument("--chunk_idx", default=1, type=int)
    args = parser.parse_args()
    
    print(f'Using config={args.config}')
    config = OmegaConf.load(args.config)
    
    inference(
        config=config,
        model_path=args.ckpt,
        vae_ckpt=args.vae_ckpt,
        n_chunk=args.n_chunk,
        chunk_idx=args.chunk_idx,
    )        


if __name__ == "__main__":
    main()
