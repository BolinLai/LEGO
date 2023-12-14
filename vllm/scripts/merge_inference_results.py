import os
import json


def merge_outputs(source_path, save_path):
    total = dict()
    for json_file in sorted(os.listdir(source_path)):
        with open(os.path.join(source_path, json_file), 'r') as infile:
            res = json.load(infile)
        total.update(res)

    with open(save_path, 'w') as outfile:
        json.dump(total, outfile, indent=4)


if __name__ == "__main__":
    merge_outputs(source_path="./vllm/out/Ego4D/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val", 
                  save_path="./vllm/out/Ego4D/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-val.json")
