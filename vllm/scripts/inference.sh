export PYTHONPATH=$PYTHONPATH:./vllm

# Inference on multiple images -- Ego4D
python -m llava.eval.run_llava_in_loop \
    --model-path /fsx/bolinlai/Models/llava/Ego4D/llava-llama-2-13b-chat-forecasting-finetune/checkpoint-450 \
    --image-dir /fsx/bolinlai/EgoGen/ego4d.fho/val \
    --action-label /data/home/bolinlai/Projects/Preprocess/ego4d_val.json \
    --query "How does the person properly {} that is displayed in the video frame?" \
    --save-path ./vllm/out/Ego4D/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val \
    --save-image-feature-path /fsx/bolinlai/LLaVA_Feature/ego4d.fho.rep/llava_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val \
    --save-text-feature-path /fsx/bolinlai/LLaVA_Feature/ego4d.fho.rep/llava_text_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450/val \
    --seed 42 \
    --num-chunks 5 \
    --chunk-idx 1

# Inference on multiple images -- EpicKitchens
# python -m llava.eval.run_llava_in_loop \
#     --model-path /fsx/bolinlai/Models/llava/EpicKitchen/llava-llama-2-13b-chat-forecasting-finetune \
#     --image-dir /fsx/bolinlai/EgoGen/epickitchen/val \
#     --action-label /data/home/bolinlai/Projects/Preprocess/epickitchen_val.json \
#     --query "How does the person properly {} that is displayed in the video frame?" \
#     --save-path ./vllm/out/EpicKitchen/llava-llama-2-13b-chat-forecasting-finetune-lastckpt/val \
#     --save-image-feature-path /fsx/bolinlai/LLaVA_Feature/epickitchen/llava_image_feature/llava-llama-2-13b-chat-forecasting-finetune/val \
#     --save-text-feature-path /fsx/bolinlai/LLaVA_Feature/epickitchen/llava_text_feature/llava-llama-2-13b-chat-forecasting-finetune/val \
#     --seed 42 \
#     --num-chunks 5 \
#     --chunk-idx 1
