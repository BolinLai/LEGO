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
    merge_outputs(source_path="/fsx-project/bolinlai/Release/vllm_output/epickitchen/llava-llama-2-13b-chat-forecasting-finetune/val", 
                  save_path="/fsx-project/bolinlai/Release/vllm_output/epickitchen/llava-llama-2-13b-chat-forecasting-finetune/llava-llama-2-13b-chat-forecasting-finetune-val.json")
