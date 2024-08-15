import os
import time
import sys
sys.path.append("./")

import json
import torch
import transformers
import torchvision.transforms.functional as F
import egovlp.model as module_arch

from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_fid import fid_score
from clip_similarity import ClipSimilarity
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.transforms.functional import InterpolationMode
from torchmetrics import PSNR, LPIPS
from egovlp.model import sim_matrix
from blip.blip import blip_decoder
from pytorch_lightning import seed_everything


def download_weights_for_metrics():
    url = "https://www.dropbox.com/scl/fi/b8gl1w5eotn498yn3tdjl/metric_pretrained.zip?rlkey=gjrj0izhycmj1imloeh3nsv8r&st=ilmbf8he&dl=1"

    exists = {
        'jx_vit_base_p16_224-80ecf9dd.pth': False,
        'distilbert-base-uncased': False,
        'egovlp.pth': False, 
        'epic_mir_plus.pth': False,
        'model_base_caption_capfilt_large.pth': False,
        'model_large_caption.pth': False
    }
    
    if os.path.exists('metrics/egovlp/pretrained/jx_vit_base_p16_224-80ecf9dd.pth'):
        exists['jx_vit_base_p16_224-80ecf9dd.pth'] = True
    if os.path.exists('metrics/egovlp/pretrained/distilbert-base-uncased'):
        exists['distilbert-base-uncased'] = True
    if os.path.exists('metrics/egovlp/pretrained/egovlp.pth'):
        exists['egovlp.pth'] = True
    if os.path.exists('metrics/egovlp/pretrained/epic_mir_plus.pth'):
        exists['epic_mir_plus.pth'] = True
    if os.path.exists('metrics/blip/pretrained/model_base_caption_capfilt_large.pth'):
        exists['model_base_caption_capfilt_large.pth'] = True
    if os.path.exists('metrics/blip/pretrained/model_large_caption.pth'):
        exists['model_large_caption.pth'] = True
    
    download = not (exists['jx_vit_base_p16_224-80ecf9dd.pth'] and exists['distilbert-base-uncased'] and exists['egovlp.pth'] \
            and exists['epic_mir_plus.pth'] and exists['model_base_caption_capfilt_large.pth'] and exists['model_large_caption.pth'])
    
    if download is True:
        print('Model weights for metrics are missing. Start downloading...')
        commands = [
            f'wget -O tmp/metric_pretrained.zip "{url}"',
            'unzip tmp/metric_pretrained.zip'
        ]
        if exists['jx_vit_base_p16_224-80ecf9dd.pth'] is False:
            commands.append('mv tmp/metric_pretrained/jx_vit_base_p16_224-80ecf9dd.pth metrics/egovlp/pretrained/')
        if exists['distilbert-base-uncased'] is False:
            commands.append('mv tmp/metric_pretrained/distilbert-base-uncased metrics/egovlp/pretrained/')
        if exists['egovlp.pth'] is False:
            commands.append('mv tmp/metric_pretrained/egovlp.pth metrics/egovlp/pretrained/')
        if exists['epic_mir_plus.pth'] is False:
            commands.append('mv tmp/metric_pretrained/epic_mir_plus.pth metrics/egovlp/pretrained/')
        if exists['model_base_caption_capfilt_large.pth'] is False:
            commands.append('mv tmp/metric_pretrained/model_base_caption_capfilt_large.pth metrics/blip/pretrained/')
        if exists['model_large_caption.pth'] is False:
            commands.append('mv tmp/metric_pretrained/model_large_caption.pth metrics/blip/pretrained/')
        
        for cmd in commands:
            os.system(cmd)
    
        os.system('rm -rf ./tmp')


egovlp_args = {
    'ego4d':{
        "video_params": {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 4,
            "pretrained": True,
            "time_init": "zeros"
        },
        "text_params": {
            "model": "distilbert-base-uncased",
            "pretrained": True,
            "input": "text"
        },
        "projection": "minimal",
        "load_checkpoint" : "metrics/egovlp/pretrained/egovlp.pth"
    },
    
    'epickitchen': {
        "video_params": {
            "model": "SpaceTimeTransformer",
            "arch_config": "base_patch16_224",
            "num_frames": 16,
            "pretrained": True,
            "time_init": "zeros"
        },
        "text_params": {
            "model": "distilbert-base-uncased",
            "pretrained": True,
            "input": "text"
        },
        "projection": "minimal",
        "load_checkpoint" : "metrics/egovlp/pretrained/epic_mir_plus.pth"
    }
}


def compute_fid(gt_path, gen_path):
    fid = fid_score.main(paths=[gt_path, gen_path])
    return fid


def compute_psnr(dataset, gt_path, gen_path):
    psnr = PSNR(data_range=1.0)

    score_sum, item_count = 0, 0
    for clip_id in tqdm(os.listdir(gt_path)):
        for action_id in os.listdir(os.path.join(gt_path, clip_id)):
            for frame in os.listdir(os.path.join(gt_path, clip_id, action_id)):
                assert os.path.exists(os.path.join(gen_path, clip_id, action_id, frame))
                gt = Image.open(os.path.join(gt_path, clip_id, action_id, frame))
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame))
                gt = transforms.ToTensor()(gt)  # values in range of [0, 1]
                gen = transforms.ToTensor()(gen)  # values in range of [0, 1]
                
                if dataset == 'epickitchen':
                    if gt.size() != gen.size():  # In epickitchen, there could be a minor difference in the dimension like (3, 256, 456) vs. (3, 256, 455)
                        gen = F.resize(gen, gt.shape[1:])

                score = psnr(gt[None], gen[None])
                score_sum += float(score.numpy())
                item_count += 1
    
    return score_sum / item_count


def compute_lpips(dataset, gt_path, gen_path):
    lpips_squeeze = LPIPS(net_type='squeeze', reduction='mean').cuda()
    lpips_vgg = LPIPS(net_type='vgg', reduction='mean').cuda()
    lpips_alex = LPIPS(net_type='alex', reduction='mean').cuda()

    score_squeeze_sum, score_vgg_sum, score_alex_sum, item_count = 0, 0, 0, 0
    for clip_id in tqdm(os.listdir(gt_path)):
        for action_id in os.listdir(os.path.join(gt_path, clip_id)):
            for frame in os.listdir(os.path.join(gt_path, clip_id, action_id)):
                assert os.path.exists(os.path.join(gen_path, clip_id, action_id, frame))
                gt = Image.open(os.path.join(gt_path, clip_id, action_id, frame))
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame))
                gt = transforms.ToTensor()(gt)  # values in range of [0, 1]
                gen = transforms.ToTensor()(gen)  # values in range of [0, 1]
                gt = gt.cuda()
                gen = gen.cuda()
                
                if dataset == 'epickitchen':
                    if gt.size() != gen.size():  # In epickitchens, there could be a minor difference in the dimension like (3, 256, 456) vs. (3, 256, 455)
                        gen = F.resize(gen, gt.shape[1:])

                score_squeeze = lpips_squeeze(gt[None], gen[None])
                score_vgg = lpips_vgg(gt[None], gen[None])
                score_alex = lpips_alex(gt[None], gen[None])
                score_squeeze_sum += float(score_squeeze.detach().cpu().numpy())
                score_vgg_sum += float(score_vgg.detach().cpu().numpy())
                score_alex_sum += float(score_alex.detach().cpu().numpy())
                item_count += 1
    
    return score_squeeze_sum / item_count, score_vgg_sum / item_count, score_alex_sum / item_count


def compute_clip_score(gt_path, gen_path):
    clip_similarity = ClipSimilarity().cuda()
    count = 0
    avg_sim_image = 0

    for clip_id in tqdm(os.listdir(gt_path)):
        for action_id in os.listdir(os.path.join(gt_path, clip_id)):
            for frame in os.listdir(os.path.join(gt_path, clip_id, action_id)):
                assert os.path.exists(os.path.join(gen_path, clip_id, action_id, frame))
                gt = Image.open(os.path.join(gt_path, clip_id, action_id, frame))
                gt = F.to_tensor(gt).float().cuda()  # values in range of [0, 1]
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame))
                gen = F.to_tensor(gen).float().cuda()  # values in range of [0, 1]

                sim_image = clip_similarity(gt[None], gen[None])
                sim_image = sim_image.cpu().numpy().tolist()[0]
                avg_sim_image += sim_image
                count += 1

    avg_sim_image /= count
    
    return avg_sim_image


# Codes are adapted from EgoVLP official codes: https://github.com/showlab/EgoVLP
def run_egovlp(dataset, model, image_0, image_1, pre_image=None):
    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        NormalizeVideo(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    pre_image = transforms.ToTensor()(pre_image) if pre_image is not None else None
    image_0 = transforms.ToTensor()(image_0)
    image_1 = transforms.ToTensor()(image_1)

    if dataset == 'epickitchen' and pre_image is not None:
        if pre_image.size() != image_1.size():  # In epickitchens, there could be a minor difference in the dimension like (3, 256, 456) vs. (3, 256, 455)
            image_1 = F.resize(image_1, pre_image.shape[1:])
    
    if dataset == 'ego4d':
        images_0 = torch.stack([image_0, image_0, image_0, image_0], dim=0) if pre_image is None else torch.stack([pre_image, pre_image, image_0, image_0], dim=0)
    elif dataset == 'epickitchen':
        images_0 = torch.stack([image_0] * 16, dim=0) if pre_image is None else torch.stack([pre_image] * 8 + [image_0] * 8, dim=0)
    else:
        raise NotImplementedError
    images_0 = images_0.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
    images_0 = image_transforms(images_0)
    images_0 = images_0.transpose(0, 1)  # recover
    images_0 = images_0.unsqueeze(0)

    if dataset == 'ego4d':
        images_1 = torch.stack([image_1, image_1, image_1, image_1], dim=0) if pre_image is None else torch.stack([pre_image, pre_image, image_1, image_1], dim=0)
    elif dataset == 'epickitchen':
        images_1 = torch.stack([image_1] * 16, dim=0) if pre_image is None else torch.stack([pre_image] * 8 + [image_1] * 8, dim=0)
    else:
        raise NotImplementedError
    images_1 = images_1.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
    images_1 = image_transforms(images_1)
    images_1 = images_1.transpose(0, 1)  # recover
    images_1 = images_1.unsqueeze(0)

    images_0 = images_0.cuda()
    images_1 = images_1.cuda()
    images_0_features = model({'video': images_0}, video_only=True, return_embeds=True)
    images_1_features = model({'video': images_1}, video_only=True, return_embeds=True)
    
    sim_image = sim_matrix(images_0_features, images_1_features)

    return sim_image


def compute_egovlp(dataset, gt_path, gen_path, use_context=False):
    assert dataset in egovlp_args, 'NotImplementedError'
    module_args = egovlp_args[dataset]
    model = getattr(module_arch, "FrozenInTime")(**module_args)
    model = model.cuda()
    model.eval().requires_grad_(False)

    count = 0
    avg_sim_image = 0

    for clip_id in tqdm(os.listdir(gt_path)):
        for action_id in os.listdir(os.path.join(gt_path, clip_id)):
            for frame in os.listdir(os.path.join(gt_path, clip_id, action_id)):
                assert os.path.exists(os.path.join(gen_path, clip_id, action_id, frame))
                gt = Image.open(os.path.join(gt_path, clip_id, action_id, frame))
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame))

                # load first image
                if use_context:
                    pre_frame = sorted(os.listdir(os.path.join(gt_path.replace('val_gt_for_metric', 'val'), clip_id, action_id)))[0]
                    assert frame != pre_frame
                    pre_image = Image.open(os.path.join(gt_path.replace('val_gt_for_metric', 'val'), clip_id, action_id, pre_frame))
                else:
                    pre_image = None
                
                sim_image = run_egovlp(dataset, model, gt, gen, pre_image=pre_image)
                sim_image = sim_image.cpu().numpy().tolist()[0][0]
                avg_sim_image += sim_image
                count += 1

    avg_sim_image /= count

    return avg_sim_image


# Codes are adapted from BLIP official codes: https://github.com/salesforce/BLIP
def compute_blip_caption(gen_path, model_type, edits):
    if model_type == 'base':
        config = {
            'pretrained': 'metrics/blip/pretrained/model_base_caption_capfilt_large.pth',
            'med_config': 'metrics/blip/med_config.json',
            'image_size': 384, 'vit': 'base', 'vit_grad_ckpt': False, 'vit_ckpt_layer': 0, 'prompt': 'a picture of',
            'num_beams': 3, 'max_length': 20, 'min_length': 5
        }
    elif model_type == 'large':
        config = {
            'pretrained': 'metrics/blip/pretrained/model_large_caption.pth',
            'med_config': 'metrics/blip/med_config.json',
            'image_size': 384, 'vit': 'large', 'vit_grad_ckpt': True, 'vit_ckpt_layer': 5, 'prompt': 'a picture of',
            'num_beams': 3, 'max_length': 20, 'min_length': 5
        }
    else:
        raise NotImplementedError

    model = blip_decoder(pretrained=config['pretrained'], med_config=config['med_config'], image_size=config['image_size'], 
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], prompt=config['prompt'])

    model = model.cuda()

    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    results = dict()
    actions, gen_batch = list(), list()
    for clip_id in tqdm(edits):  # Note that the order of inference is very important! Different order results in different captions.
        results[clip_id] = dict()
        for action_id in edits[clip_id]:  # Note that the order of inference is very important! Different order results in different captions.
            for frame in os.listdir(os.path.join(gen_path, clip_id, action_id)):
                gen = Image.open(os.path.join(gen_path, clip_id, action_id, frame)).convert('RGB')
                gen = transform_test(gen)
                gen = gen.cuda()
                actions.append((clip_id, action_id))
                gen_batch.append(gen)

                if len(actions) == 30:
                    gen_batch = torch.stack(gen_batch, dim=0)
                    captions = model.generate(gen_batch, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
                    for act, cap in zip(actions, captions):
                        results[act[0]][act[1]] = cap
                    gen_batch = list()
                    actions.clear()
    
    if len(gen_batch) > 0:
        gen_batch = torch.stack(gen_batch, dim=0) if len(gen_batch) > 1 else gen_batch[0][None]
        captions = model.generate(gen_batch, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], min_length=config['min_length'])
        for act, cap in zip(actions, captions):
            results[act[0]][act[1]] = cap
    
    save_path = os.path.join(os.path.dirname(gen_path), 'metric/blip.json' if model_type == 'base' else 'metric/blip-l.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)


def compute_text_similarity(dataset, edits, blip_path, llava_key):
    assert dataset in egovlp_args, 'NotImplementedError'
    module_args = egovlp_args[dataset]
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='metrics/egovlp/pretrained/distilbert-base-uncased', TOKENIZERS_PARALLELISM=False)
    egovlp = getattr(module_arch, "FrozenInTime")(**module_args)
    egovlp = egovlp.cuda()
    egovlp.eval().requires_grad_(False)

    count = 0
    avg_egovlp_sim_desc = 0

    with open(blip_path, 'r') as infile:
        blip_res = json.load(infile)
    
    for clip_id in tqdm(edits):
        for action_id in edits[clip_id]:
            gt_desc = edits[clip_id][action_id][llava_key]
            blip_narration = blip_res[clip_id][action_id]
            
            token_desc = tokenizer(gt_desc, return_tensors='pt', padding=True, truncation=True)
            token_desc = {key: val.cuda() for key, val in token_desc.items()}
            token_blip = tokenizer(blip_narration, return_tensors='pt', padding=True, truncation=True)
            token_blip = {key: val.cuda() for key, val in token_blip.items()}
            gt_egovlp_desc_embed = egovlp.compute_text(token_desc)
            blip_egovlp_embed = egovlp.compute_text(token_blip)
            egovlp_sim_desc = torch.nn.functional.cosine_similarity(gt_egovlp_desc_embed, blip_egovlp_embed)
            egovlp_sim_desc = egovlp_sim_desc.cpu().numpy().tolist()[0]

            avg_egovlp_sim_desc += egovlp_sim_desc
            count += 1
    
    avg_egovlp_sim_desc /= count

    return avg_egovlp_sim_desc


def compute_metrics(dataset, gt_path, gen_path, edit_file, llava_key, cmpt_fid=True, cmpt_psnr=True, cmpt_lpips=True, 
                    cmpt_clip=True, cmpt_vlp=True, cmpt_vlp_plus=True, cmpt_blip_base=True, cmpt_blip_large=True, save_res=True):
    
    with open(edit_file, 'r') as infile:
        edits = json.load(infile)

    print(f'Computing metrics for {gen_path}')
    
    content = dict()

    if cmpt_fid:
        print('Computing FID')
        fid = compute_fid(gt_path, gen_path)
        print('\nFID:', fid)
        content['fid'] = fid

    if cmpt_psnr:
        print('Computing PSNR')
        psnr = compute_psnr(dataset, gt_path, gen_path)
        print('\nPSNR:', psnr)
        content['psnr'] = psnr

    if cmpt_lpips:
        print('Computing LPIPS')
        lpips_squeeze, lpips_vgg, lpips_alex = compute_lpips(dataset, gt_path, gen_path)
        print('\nLPIPS (SENet):', lpips_squeeze)
        print('LPIPS (VGG):', lpips_vgg)
        print('LPIPS (ALEX):', lpips_alex)
        content['lpips_squeeze'] = lpips_squeeze
        content['lpips_vgg'] = lpips_vgg
        content['lpips_alex'] = lpips_alex

    if cmpt_clip:
        print('Computing CLIP')
        sim_image = compute_clip_score(gt_path, gen_path)
        print('ClipSimilarity of Gen and GT:', sim_image)
        content['clip_image_sim'] = sim_image

    if cmpt_vlp:
        print('Computing EgoVLP')
        ego_sim_image = compute_egovlp(dataset, gt_path, gen_path, use_context=False)
        print('EgoSimilarity of Gen and GT:', ego_sim_image)
        content['egovlp_image_sim'] = ego_sim_image

    if cmpt_vlp_plus:
        print('Computing EgoVLP+')
        ego_plus_sim_image = compute_egovlp(dataset, gt_path, gen_path, use_context=True)
        print('EgoSimilarity+ of Gen and GT:', ego_plus_sim_image)
        content['egovlp_plus_image_sim'] = ego_plus_sim_image

    if cmpt_blip_base:
        print('Computing BLIP-B Similarity')
        compute_blip_caption(gen_path, model_type='base', edits=edits)
        egovlp_sim_desc = compute_text_similarity(dataset, edits, blip_path=os.path.join(os.path.dirname(gen_path), 'metric/blip.json'), llava_key=llava_key)
        print('BLIP-B EgoVLP Similarity with Description:', egovlp_sim_desc)
        content['blip_base_egovlp_sim'] = egovlp_sim_desc

    if cmpt_blip_large:
        print('Computing BLIP-L Similarity')
        compute_blip_caption(gen_path, model_type='large', edits=edits)
        egovlp_sim_desc = compute_text_similarity(dataset, edits, blip_path=os.path.join(os.path.dirname(gen_path), 'metric/blip-l.json'), llava_key=llava_key)
        print('BLIP-L EgoVLP Similarity with Description:', egovlp_sim_desc)
        content['blip_large_egovlp_sim'] = egovlp_sim_desc

    print('All metrics:')
    print(content)
    
    if save_res:
        save_path = os.path.join(os.path.dirname(gen_path), 'metric/metric.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.exists(save_path):
            with open(save_path, 'r') as infile:
                save = json.load(infile)
        else:
            save = dict()
        save.update(content)
        with open(save_path, 'w') as outfile:
            json.dump(save, outfile, indent=4)



if __name__ == '__main__':
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="ego4d or epickitchen")
    parser.add_argument("--gen_path", type=str, help="Path to generated images, e.g. /fsx/bolinlai/Models/instruct_pix2pix/train_ego4d_llava_txt_mid/analysis/epoch=000119-step=000009999-e=0-s=100-si=1.5/images")
    parser.add_argument("--llava_key", type=str, default='llava_forecast_finetune')
    parser.add_argument("--gt_path", type=str, help="Path to val_for_metric")
    parser.add_argument("--edit_file", type=str, help="Path to action label")
    args = parser.parse_args()

    assert args.dataset in ['ego4d', 'epickitchen']
    # gt_path = '/fsx/bolinlai/EgoGen/ego4d.fho/val_gt_for_metric' if args.dataset == 'ego4d' else '/fsx/bolinlai/EgoGen/epickitchen/val_gt_for_metric'  # [Change to your path of GT]
    # edit_file = '/data/home/bolinlai/Projects/Preprocess/ego4d_val.json' if args.dataset == 'ego4d' else '/data/home/bolinlai/Projects/Preprocess/epickitchen_val.json'  # [Change to your path of the editing file]

    print(f'Running metrics for {args.dataset}.')
    print(f'GT path is {args.gt_path}.')
    print(f'Edit file is {args.edit_file}.')
    print(f'LLaVA key is {args.llava_key}')

    seed_everything(42)

    compute_metrics(
        dataset=args.dataset,
        gt_path=args.gt_path,
        gen_path=args.gen_path,
        edit_file=args.edit_file,
        llava_key=args.llava_key,
        cmpt_fid=True,
        cmpt_psnr=True,
        cmpt_lpips=True,
        cmpt_clip=True,
        cmpt_vlp=True,
        cmpt_vlp_plus=True,
        cmpt_blip_base=True,
        cmpt_blip_large=True,
        save_res=True
    )

    print(time.time() - start)
