import argparse
import torch
import math
import os
import json
import nltk
import random
import numpy as np
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer


def seed_everything(seed):
    """
    Copied from pytorch_lightning.seed.seed_everything
    """
    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = "0"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    assert args.setting in ['description', 'forecast'], 'Please set --setting as "description" or "forecast".'

    if 'ego4d' in args.action_label.rsplit('/', 1)[-1]:
        dataset = 'ego4d'
    elif 'epickitchen' in args.action_label.rsplit('/', 1)[-1]:
        dataset = 'epickitchen'
    else:
        raise NotImplemented

    print(f"Using dataset {dataset}")
    print(f"Infering {args.chunk_idx}/{args.num_chunks}")

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # Add 'llava' in the model name if you want to do inference without finetune ----------
    model_name += '-llava-llama-2'
    # ----------------------------------------------------------------------------------
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    with open(args.action_label, 'r') as infile:
        edits = json.load(infile)

    nltk.data.path.append('/data/home/bolinlai/Projects/nltk_data')
    res = dict()

    clips = get_chunk(list(edits.keys()), args.num_chunks, args.chunk_idx-1)
    for clip_id in clips:
        res[clip_id] = dict()
        for action_id in edits[clip_id].keys():
            image_path = edits[clip_id][action_id]['image_0'] if args.setting == 'forecast' else edits[clip_id][action_id]['image_1']
            image = load_image(os.path.join(args.image_dir, image_path))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            if dataset == 'ego4d':
                action = edits[clip_id][action_id]['action']
                action = action[2:].strip()  # remove #C at the beginning
                action = action[2:].strip()  # remove C
                split = action.split(' ', 1)
                if len(split) == 2:
                    verb, remain = split
                else:  # sometimes there's only a verb in the action
                    verb = split[0]
                    remain = ''
                verb = WordNetLemmatizer().lemmatize(verb.lower(), 'v')
                if remain != '':
                    action = verb + ' ' + remain
                else:
                    action = verb
            
            elif dataset == 'epickitchen':
                action = edits[clip_id][action_id]['action']

            else:
                raise NotImplemented

            inp = args.query.format(action)

            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles

            print(f"{roles[1]}: ", end="")

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            if args.save_image_feature_path is not None:
                model.save_image_feature_path = args.save_image_feature_path
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    image_path=image_path,
                    return_dict_in_generate=False if args.save_text_feature_path is None else True,
                    output_hidden_states=False if args.save_text_feature_path is None else True,)

            if isinstance(output_ids, dict):
                outputs = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:]).strip()
            else:
                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            outputs = outputs.replace('</s>', '').strip()  # remove the end token of stream
            res[clip_id][action_id] = {'original action': edits[clip_id][action_id]['action'], 'action': action, 'description': outputs}

            # save llava text features
            if args.save_text_feature_path is not None:
                tmp = list()
                for w in range(len(output_ids['hidden_states'])):
                    txt_feature = torch.cat(output_ids['hidden_states'][w], dim=0)  # (41, n, 5120)
                    txt_feature = txt_feature[-1, -1, :]  # (5120,)
                    tmp.append(txt_feature)
                llava_text_feature = torch.stack(tmp, dim=0)
                save_text_feature_path = f'{args.save_text_feature_path}/{image_path.replace("jpg", "npy")}'
                if not os.path.exists(save_text_feature_path):
                    os.makedirs(os.path.dirname(save_text_feature_path), exist_ok=True)
                    np.save(save_text_feature_path, llava_text_feature.cpu().numpy().squeeze())
                
        # save llava output
        os.makedirs(args.save_path, exist_ok=True)
        save_path = os.path.join(args.save_path, f"{args.save_path.split('/')[-2]}_{args.chunk_idx}.json")
        with open(save_path, 'w') as outfile:
            json.dump(res, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--action-label", type=str, required=True)
    parser.add_argument("--setting", default='forecast', type=str)
    parser.add_argument("--save-path", type=str, required=True, help='Path to save llava results.')
    parser.add_argument("--save-image-feature-path", default=None, type=str, help='Path to save llava image feature.')
    parser.add_argument("--save-text-feature-path", default=None, type=str, help='Path to save llava text feature.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunk-idx", type=int, required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    eval_model(args)
