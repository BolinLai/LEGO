#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
# PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-13b-chat"
################## LLaMA-2 ##################

# Finetune LLM from scratch - description
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero3_offload.json \
#     --model_name_or_path /fsx/bolinlai/Models/Pretrained/llama/models--meta-llama--Llama-2-13b-hf/snapshots/a5a274e267651cf851f59ed47a4eab85640cdcc9 \
#     --version $PROMPT_VERSION \
#     --data_path /data/home/bolinlai/Projects/Preprocess/epickitchen_for_tuning_description.json \
#     --image_folder /fsx/bolinlai/EgoGen/epickitchen/train \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --pretrain_mm_mlp_adapter /checkpoints/bolinlai/llava/released/models--liuhaotian--llava-pretrain-llama-2-13b-chat/snapshots/baf3e7ec5f3636aa9923aea235a0a6177a244b8c/mm_projector.bin \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /fsx/bolinlai/Models/llava/EpicKitchen/llava-$MODEL_VERSION-description-scratch \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 140 \
#     --save_total_limit 4 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


# Finetune LLM from scratch - forecasting
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero3_offload.json \
#     --model_name_or_path /fsx/bolinlai/Models/Pretrained/llama/models--meta-llama--Llama-2-13b-hf/snapshots/a5a274e267651cf851f59ed47a4eab85640cdcc9 \
#     --version $PROMPT_VERSION \
#     --data_path /data/home/bolinlai/Projects/Preprocess/epickitchen_for_tuning_forecasting.json \
#     --image_folder /fsx/bolinlai/EgoGen/epickitchen/train \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --pretrain_mm_mlp_adapter /checkpoints/bolinlai/llava/released/models--liuhaotian--llava-pretrain-llama-2-13b-chat/snapshots/baf3e7ec5f3636aa9923aea235a0a6177a244b8c/mm_projector.bin \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /fsx/bolinlai/Models/llava/EpicKitchen/llava-$MODEL_VERSION-forecasting-scratch \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 140 \
#     --save_total_limit 4 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


# Finetune LLM from released checkpoint - description
# deepspeed llava/train/train_mem.py \
#     --deepspeed scripts/zero3_offload.json \
#     --model_name_or_path /checkpoints/bolinlai/llava/released/models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520 \
#     --version $PROMPT_VERSION \
#     --data_path /data/home/bolinlai/Projects/Preprocess/epickitchen_for_tuning_description.json \
#     --image_folder /fsx/bolinlai/EgoGen/epickitchen/train \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /fsx/bolinlai/Models/llava/EpicKitchen/llava-$MODEL_VERSION-description-finetune \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 140 \
#     --save_total_limit 4 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb


# Finetune LLM from released checkpoint - forecasting
deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero3_offload.json \
    --model_name_or_path /checkpoints/bolinlai/llava/released/models--liuhaotian--llava-llama-2-13b-chat-lightning-preview/snapshots/bcda0227a7f371117a195ef0af88c1616a830520 \
    --version $PROMPT_VERSION \
    --data_path /data/home/bolinlai/Projects/Preprocess/epickitchen_for_tuning_forecasting.json \
    --image_folder /fsx/bolinlai/EgoGen/epickitchen/train \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /fsx/bolinlai/Models/llava/EpicKitchen/llava-$MODEL_VERSION-forecasting-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 140 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
