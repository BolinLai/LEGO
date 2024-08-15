# LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

### ECCV 2024 (Oral)

### [Project Page](https://bolinlai.github.io/Lego_EgoActGen/) | [Paper](https://arxiv.org/pdf/2312.03849) | [Dataset](https://www.dropbox.com/scl/fo/4m0v9oy753aimas8rz6v1/ANoJhZQz2BdcGIVLzUsHdP0?rlkey=o8saklcszfc098mjnpid767ic&dl=0)

### Update

[08/15] Our work was selected as **oral presentation** in ECCV 2024.

[08/15] We have released our model weights and completed README with detailed guidance.

[07/21] <font color=red>**Our dataset has been released!**</font>

<!-- Thank you for your interest in our work! The first version of the code has been released. We are editing the README instructions.

ToDo:

- [x] Dataset

- [x] Codes

- [x] README

- [x] Checkpoints -->

 <img src='https://bolinlai.github.io/Lego_EgoActGen/figures/visualization_new_actions.png'/>



## Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [GPT-Curated Data (Optional)](#gpt-curated-data-optional)
- [VLLM Features](#vllm-features)
- [Model Weights](#model-weights)
- [Train and Inference](#train-and-inference)
  - [VLLM Training](#vllm-training)
  - [VLLM Inference](#vllm-inference)
  - [LDM Training](#ldm-training)
  - [LDM Inference](#ldm-inference)
- [Metrics](#metrics)
- [BibTex](#bibtex)
- [Acknowledge](#acknowledgement)



## Setup

Due to the incompatibility of VLLM and LDM packages, we use two environments for each model.

Install the dependencies for VLLM.

```shell
conda env create -f vllm_env.yaml  # The env name is "vllm".
conda activate vllm
pip install flash-attn==2.0.7 --no-build-isolation
```

Install the dependencies for LDM.

```shell
conda env create -f ldm_env.yaml  # The env name is "ldm".
```


## Dataset

Run the command below to download our dataset, or download via this [link](https://www.dropbox.com/scl/fo/4m0v9oy753aimas8rz6v1/ANoJhZQz2BdcGIVLzUsHdP0?rlkey=o8saklcszfc098mjnpid767ic&dl=0) and unzip `EgoGen.zip` to your local path. .
```shell
bash scripts/download_dataset.sh [your_local_path]  # replace [your_local_path] to your local download path.
```
Our dataset is composed of video frames (in `EgoGen.zip`) and action labels/descriptions (in `*.json`) from Ego4D and Epic-Kitchens.

The structure of the dataset is as follows. Note that `val_gt_for_metric` only contains ground truth images of `val`. It's simply used for metric calculation (FID score) and not invovled in training and inference.
```
[your_local_path]
        |
        |-- EgoGen
        |     |
        |     |-- ego4d.fho
        |     |       |-- train
        |     |       |-- val
        |     |       |-- val_gt_for_metric
        |     |
        |     |-- epickitchen
        |             |-- train
        |             |-- val
        |             |-- val_gt_for_metric
        |
        |-- ego4d_train.json  # ego4d image paths and actions for training
        |-- ego4d_val.json  # ego4d image paths and actions for evaluation
        |-- ego4d_metadata.json  # ego4d metadata of selected video frames
        |-- epickitchen_train.json  # epickitchens image paths and actions for training
        |-- epickitchen_val.json  # epickitchens image paths and actions for evaluation
        |-- epickitchen_metadata.json  # epickitchens metadata of selected video frames
    
```
The test sets of Ego4d and Epic-Kitchens are hidden so we use the validation set as test set in our experiments. In `ego4d_train.json`, `ego4d_val.json`, `epickitchen_train.json` and `epickitchen_val.json`, `image_0` and `image_1` denote the source and target images for action frame generation. `action` refers to the original action labels/descriptions in the Ego4D and Epic-Kitchens. We also release the enriched action descriptions (`llava_forecast_finetune`) generated by LLM. In `ego4d_metadata.json` and `epickitchen_metadata.json`, we release the meta data (e.g., fps, resolution, object bbox, etc.) of the selected video frames.


## GPT-Curated Data (Optional)

We release our detailed acton descriptions curated from GPT-3.5, which are used for VLLM instruction tuning. You can download by running the following command, or from this [link](https://www.dropbox.com/scl/fo/2iez9r5k5z9f0azxchb09/AKC_p3Y96wHpDXCOgOYnZB0?rlkey=ndh4vzedl3w66b4r90kzzz5s7&st=ozy2upb1&dl=0).
```shell
bash scripts/download_gpt_curated_data.sh
```
**Note: This step is only necessary if you want to finetune the VLLM component in LEGO. Otherwise, you can directly use the released [VLLM weights](#model-weights) for inference.**


## VLLM Features

In addition, we also release the VLLM image and text features, and then you can train the LDM component without running VLLM inference. You can download and unzip all features by running this command, or from this [link](https://www.dropbox.com/scl/fo/5tr0fjzjxzimnvgx6tath/AGQZnNWt8dHERcyBPzTdIGQ?rlkey=1gqx4p0v4pf869q225rwynt3r&dl=0).

Please make sure there are 500GB available on your machine before downloading.
```shell
bash scripts/download_vllm_features.sh [your_local_path]  # replace [your_local_path] to your local download path.
```
If you download from the link, please unzip each zip file in the same folder by using `unzip xxx.zip`. For ego4d image features, we split the data in `.zip` and `.z01` files. You can put them in the same folder and run `zip -s0 llava-llama-2-13b-chat-forecasting-finetune-ckpt450-split.zip --out merge.zip` to merge them to one single zip file. Then you can unzip `merge.zip`.


## Model Weights

|      | Ego4D | Epic-Kitchens |
|:----:|:----: | :----:        |
|VLLM  | [download](https://www.dropbox.com/scl/fi/7ielqu0joqu3ftr7r7pm7/llava-llama-2-13b-chat-forecasting-finetune.zip?rlkey=tnnhskshluoizkro1c6rrpo83&st=19yhttf3&dl=0)  |    [download](https://www.dropbox.com/scl/fi/x3but1rmnau4w05utrx0e/llava-llama-2-13b-chat-forecasting-finetune.zip?rlkey=x2yotqy0auvk5mmwppmamdx1s&st=918bxwnr&dl=0)       |
|LDM   | download  |    [download](https://www.dropbox.com/scl/fi/f6m6l774dvexy3diklal9/epickitchen_diffusion_with_vllm_feature.ckpt?rlkey=qc978bzt13v7eoh2ajm2o96da&st=vxmawpr9&dl=0)       |


## Train and Inference

We train the VLLM and LDM components separately. Both of them are trained on 8x40G A100.

**Note: As a quick start, you can directly use our released enriched action descriptions (in our [dataset](#dataset)) and [VLLM features](#vllm-features), to skip VLLM instruction tuning and inference. Then you can jump to [LDM Training](#ldm-training).**

### VLLM Training

#### Preparation

Activate `vllm` virtual environment.

```shell
conda activate vllm
```

Download pretrained llava weights by

```shell
wget -O [your_path]/llava_pretrained.zip "https://www.dropbox.com/scl/fi/q5yy8znjirymfe9kte2a2/llava_pretrained.zip?rlkey=qbskcxd85qxg5jphb50lvxd4a&st=qwdxpg2o&dl=1"
unzip [your_path]/llava_pretrained.zip -d [your_path]
rm [your_path]/llava_pretrained.zip
```

Before running the script, you have to update the paths of dataset and pretrained weights in `vllm/scripts/finetune_ego4d.sh` (for training on Ego4D) and `vllm/scripts/finetune_epickitchen.sh` (for training on Epic-Kitchens) to your local paths.

`--model_name_or_path`: The path of pretrained VLLM checkpoint for initialization.

`--data_path`: The path of detailed action descriptions [curated from GPT-3.5](#gpt-curated-data-optional).

`--image_folder`: The path of Ego4D/Epic-Kitchens training data (i.e., video frames).

`--output_dir`: The path to save checkpoints. 

#### Train VLLM on Ego4D

Then run the command below.

```shell
bash vllm/scripts/finetune_ego4d.sh
```

#### Train VLLM on Epic-Kitchens

Then run the command below.

```shell
bash vllm/scripts/finetune_epickitchen.sh
```

---

### VLLM Inference

#### Preparation

Activate `vllm` virtual environment.

```shell
conda activate vllm
```

To speed up inference, we divide the data into 5 chunks and run inference on them separately. There are two ways to run inference on **Ego4D** or **Epic-Kitchens**.

#### (1) Use Slurm

If you are using slurm to launch jobs, before running the script, you have to update the paths in `vllm/scripts/sbatch_inference.sh` to your local paths.

`model_path`: The path of [instruction-tuned VLLM weights](#model-weights).

`image_dir`: The path of video frames.

`action_label`: The path of action labels (i.e., *.json files downloaded with video frames).

`save_path`: The path to save generated enriched action descriptions.

`save_image_feature_path`: The path to save VLLM image features.

`save_text_feature_path`: The path to save VLLM text features.

The configuration of slurm can be edited in `vllm/scripts/sbatch_inference/inference_sbatch_*.sh`. Then run the command below and check the logs in `vllm/out/logs`.

```shell
bash vllm/scripts/sbatch_inference.sh
```

Then merge the output in one json file using

```python
python vllm/scripts/merge_inference_results.py
```

#### (2) Without Slurm

You need to manually run the inference on each chunk. You can use `--num-chunks` to control how many chunks the data will be divided into, and `--chunk-idx` to specify which chunk to be used for inference (e.g., `--num-chunks=5  --chunk-idx=3` means dividing data into 5 chunks and run inference on the third chunk). The paths should be changed to your local paths as elaborated above.

```shell
export PYTHONPATH=$PYTHONPATH:./vllm

python -m llava.eval.run_llava_in_loop \
    --model-path /fsx-project/bolinlai/Release/checkpoints/VLLM/ego4d/llava-llama-2-13b-chat-forecasting-finetune \
    --image-dir /fsx-project/bolinlai/Release/dataset/EgoGen/ego4d.fho/val \
    --action-label /fsx-project/bolinlai/Release/dataset/ego4d_val.json \
    --query "How does the person properly {} that is displayed in the video frame?" \
    --save-path /fsx-project/bolinlai/Release/vllm_output/ego4d/llava-llama-2-13b-chat-forecasting-finetune/val \
    --save-image-feature-path /fsx-project/bolinlai/Release/vllm_features/ego4d/vllm_image_features/llava-llama-2-13b-chat-forecasting-finetune/val \
    --save-text-feature-path /fsx-project/bolinlai/Release/vllm_features/ego4d/vllm_text_features/llava-llama-2-13b-chat-forecasting-finetune/val \
    --seed 42 \
    --num-chunks 5 \
    --chunk-idx 1
```

Then merge the output in one json file using

```python
python vllm/scripts/merge_inference_results.py
```

---

### LDM Training

#### Preparation

Activate `ldm` virtual environment.

```shell
conda activate ldm
```

Download pretrained stable diffusion weights by

```shell
wget -O [your_path]/stable_diffusion.zip "https://www.dropbox.com/scl/fi/773bpwnb2m4db2uvo0d64/stable_diffusion.zip?rlkey=qgk8mg5j4hrqqbsxkz0gt0os7&st=b5wltovy&dl=1"
unzip [your_path]/stable-diffusion.zip -d [your_path]
rm [your_path]/stable-diffusion.zip
```

Before launching the job, you have to update the paths in `configs/train_ego4d.yaml` and `configs/train_epickitchen.yaml` for training on Ego4D and Epic-Kitchens, respectively.

`model.params.ckpt_path`: The path of pretrained latent diffusion model weights for initialization.

`data.params.train.params.data_path`: The path of video frames in training set.

`data.params.train.params.edit_path`: The path of action descriptions in training set (i.e., `ego4d_train.json/epickitchen_train.json` downloaded with video frames, or you can replace the action descriptions with the VLLM output generated by yourself in [VLLM Inference](#vllm-inference)).

`data.params.train.params.additional_cond_path`: The paths of VLLM image and text features in training set (our released features, or features generated by yourself in [VLLM Inference](#vllm-inference)).

`data.params.validation.params.data_path`: The path of video frames in val set.

`data.params.validation.params.edit_path`: The path of action descriptions in val set (i.e., `ego4d_val.json/epickitchen_val.json` downloaded with video frames, or you can replace the action descriptions with the VLLM output generated by yourself in [VLLM Inference](#vllm-inference)).

`data.params.validation.params.additional_cond_path`: The paths of VLLM image and text features in val set (our released features, or features generated by yourself in [VLLM Inference](#vllm-inference)).

#### Train LDM on Ego4D

Run the command below. The checkpoints will be saved in `logs/`.

```shell
python main.py --name lego_ego4d --base configs/train_ego4d.yaml --train --gpus 0,1,2,3,4,5,6,7
```

#### Train LDM on Epic-Kitchens

Run the command below. The checkpoints will be saved in `logs/`.

```shell
python main.py --name lego_epickitchens --base configs/train_epickitchen.yaml --train --gpus 0,1,2,3,4,5,6,7
```

---

### LDM Inference

#### Preparation

Activate `ldm` virtual environment.

```shell
conda activate ldm
```

To speed up inference, we divide the data into 8 chunks and run inference on them separately. Similar to training, you need to update the paths in `configs/generate_ego4d.yaml` and `configs/generate_epickitchen.yaml` for Ego4D and Epic-Kitchens inference, respectively.

`data.metadata_path`: The path of metadata (released in our dataset).

`data.params.data_path`: The path of video frames in val set.

`data.params.edit_path`: The path of action descriptions in val set (i.e., `ego4d_val.json/epickitchen_val.json`).

`data.params.additional_cond_path`: The paths of VLLM image and text features in val set.

#### Run LDM inference on Ego4D

(1) Use Slurm

Edit slurm configuration in `sbatch_inference/ego4d_inference/test_ego4d_sbatch_*.sh` to comply with your cluster.

Run the following command with your local path to the checkpoint and check the logs in `logs/out`. Generated images will be saved under the same directory of checkpoint.

```shell
bash test_ego4d.sh logs/ego4d_diffusion_with_vllm_feature.ckpt
```

(2) Without Slurm

You need to manually run inference on each chunk with the command below. `--n_chunk 8 --chunk_idx 1` means dividing data into 8 chunks and run inference on the first chunk.

```shell
python metrics/inference.py --config configs/generate_ego4d.yaml --ckpt logs/ego4d_diffusion_with_vllm_feature.ckpt --n_chunk 8 --chunk_idx 1
```

#### Run LDM inference on Epic-Kitchens

(1) Use Slurm

Edit slurm configuration in `sbatch_inference/epickitchen_inference/test_epickitchen_sbatch_*.sh` to comply with your cluster.

Run the following command with your local path to the checkpoint and check the logs in `logs/out`. Generated images will be saved under the same directory of checkpoint. Generated images will be saved under the same directory of checkpoint.

```shell
bash test_epickitchen.sh logs/epickitchen_diffusion_with_vllm_feature.ckpt
```

(2) Without Slurm

You need to manually run inference on each chunk with the command below. `--n_chunk 8 --chunk_idx 1` means dividing data into 8 chunks and run inference on the first chunk. Generated images will be saved under the same directory of checkpoint.

```shell
python metrics/inference.py --config configs/generate_epickitchen.yaml --ckpt logs/epickitchen_diffusion_with_vllm_feature.ckpt --n_chunk 8 --chunk_idx 1
```


## Metrics

#### Preparation

Activate `ldm` virtual environment.

```shell
conda activate ldm
```

If this is your first time to run metric calcuation, you need to download some model weights by running

```shell
bash scripts/download_metric_weights.sh
```

#### Calculate metrics on Ego4D

Replace the following path to your local path and run the command.

`--gen_path`: The path of generated action frames.

`--gt_path`: The path of `val_gt_for_metric` (downloaded with dataset).

`--edit_file`: THe path of action descriptions (i.e., `ego4d_val.json`).

```shell
python metrics/all_metrics_in_one.py --dataset ego4d --llava_key llava_forecast_finetune --gen_path logs/ego4d_diffusion_with_vllm_feature-e=0-s=100-si=1.5/images --gt_path /fsx-project/bolinlai/Release/dataset/EgoGen/ego4d.fho/val_gt_for_metric --edit_file /fsx-project/bolinlai/Release/dataset/ego4d_val.json
```

#### Calculate metrics on Epic-Kitchens

Similarly, update the paths (as above) and run the command.

```shell
python metrics/all_metrics_in_one.py --dataset epickitchen --llava_key llava_forecast_finetune --gen_path logs/epickitchen_diffusion_with_vllm_feature-e=0-s=100-si=1.5/images --gt_path /fsx-project/bolinlai/Release/dataset/EgoGen/epickitchen/val_gt_for_metric --edit_file /fsx-project/bolinlai/Release/dataset/epickitchen_val.json
```


## BibTex

If you find LEGO useful for your work, please cite using this BibTex.

```
@article{lai2023lego,
        title={LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning},
        author={Lai, Bolin and Dai, Xiaoliang and Chen, Lawrence and Pang, Guan and Rehg, James M and Liu, Miao},
        journal={arXiv preprint arXiv:2312.03849},
        year={2023}
      }
```


## Acknowledgement

Our code was built on [LLaVA](https://github.com/haotian-liu/LLaVA) and [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix). We appreciate the authors of the two awesome codebases.
