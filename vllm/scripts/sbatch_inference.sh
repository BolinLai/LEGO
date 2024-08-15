#!/bin/bash
export PYTHONPATH=$PYTHONPATH:./vllm

model_path='/fsx-project/bolinlai/Release/checkpoints/VLLM/ego4d/llava-llama-2-13b-chat-forecasting-finetune'
image_dir='/fsx-project/bolinlai/Release/dataset/EgoGen/ego4d.fho/val'
action_label='/fsx-project/bolinlai/Release/dataset/ego4d_val.json'
save_path='/fsx-project/bolinlai/Release/vllm_output/ego4d/llava-llama-2-13b-chat-forecasting-finetune/val'
save_image_feature_path='/fsx-project/bolinlai/Release/vllm_features/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune/val'
save_text_feature_path='/fsx-project/bolinlai/Release/vllm_features/ego4d/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune/val'
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
