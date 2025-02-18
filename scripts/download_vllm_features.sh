#!/bin/bash

echo Downloading VLLM features to $1 ...

mkdir -p $1/ego4d/vllm_image_feature
echo Downloading VLLM image features of Ego4D...
wget -O $1/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.z01 "https://www.dropbox.com/scl/fi/aq86iz0w5oz5exktcwwap/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.z01?rlkey=386v388ew2tbu1391vwjikcvc&st=5948z042&dl=1"
wget -O $1/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.zip "https://www.dropbox.com/scl/fi/wrw6zbwsfwx8as7xyugus/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.zip?rlkey=d3cer6xfqpvtnlaapl90xrh6c&st=hefav5dq&dl=1"

mkdir -p $1/ego4d/vllm_text_feature
echo Downloading VLLM text features of Ego4D...
wget -p $1/ego4d/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450.zip "https://www.dropbox.com/scl/fi/i9n86tu88ty3qtg3jhnh8/llava-llama-2-13b-chat-forecasting-finetune-ckpt450.zip?rlkey=03j47m7i0a75gejrf0agkwwhw&st=rzh4dmbv&dl=1"

mkdir -p $1/epickitchen/vllm_image_feature
echo Downloading VLLM image features of Epic-Kitchens...
wget -O $1/epickitchen/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune.zip "https://www.dropbox.com/scl/fi/qx9fyb0gc0regjb1xyntr/llava-llama-2-13b-chat-forecasting-finetune.zip?rlkey=8uqf9hmxdbf3307uctqyg1civ&st=6yi0nwrm&dl=1"

mkdir -p $1/epickitchen/vllm_text_feature
echo Downloading VLLM text features of Epic-Kitchens...
wget -O $1/epickitchen/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune.zip "https://www.dropbox.com/scl/fi/2c1warf3mlho0q4t7knyl/llava-llama-2-13b-chat-forecasting-finetune.zip?rlkey=9wg2s70ei512yi4cq464zpp9t&st=br4nw308&dl=1"

echo Unzipping features...
zip -s0 $1/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.zip --out $1/ego4d/llava_image_feature/merge.zip
unzip $1/ego4d/vllm_image_feature/merge.zip -d $1/ego4d/vllm_image_feature/
rm $1/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.z01
rm $1/ego4d/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.zip
rm $1/ego4d/vllm_image_feature/merge.zip

unzip $1/ego4d/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450.zip -d $1/ego4d/vllm_text_feature/
rm $1/ego4d/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune-ckpt450.zip

unzip $1/epickitchen/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune.zip -d $1/epickitchen/vllm_image_feature/
rm $1/epickitchen/vllm_image_feature/llava-llama-2-13b-chat-forecasting-finetune.zip

unzip $1/epickitchen/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune.zip -d $1/epickitchen/vllm_text_feature/
rm $1/epickitchen/vllm_text_feature/llava-llama-2-13b-chat-forecasting-finetune.zip

echo Done!
