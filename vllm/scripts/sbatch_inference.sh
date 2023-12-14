#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./vllm

model_path='/fsx/bolinlai/Models/llava/Ego4D/llava-llama-2-13b-chat-forecasting-finetune/checkpoint-450'
image_dir='/fsx/bolinlai/EgoGen/ego4d.fho/val'
action_label='/data/home/bolinlai/Projects/Preprocess/ego4d_val.json'
save_path='./vllm/out/Ego4D/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val'
save_image_feature_path='/fsx/bolinlai/LLaVA_Feature/ego4d.fho.rep/llava_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val'
save_text_feature_path='/fsx/bolinlai/LLaVA_Feature/ego4d.fho.rep/llava_text_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val'
seed=42

echo Path to checkpoint:$model_path
echo Image directory:$image_dir
echo Path to train/val file:$action_label
echo Saving path:$save_path
echo Image feature saving path:$save_image_feature_path
echo Text feature saving path:$save_text_feature_path
echo Seed:$seed

for i in 1 2 3 4 5; do
    sbatch ./vllm/scripts/sbatch_inference/inference_sbatch_$i.sh $model_path $image_dir $action_label $save_path $save_image_feature_path $save_text_feature_path $seed
done