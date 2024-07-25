# LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

### [Project Page](https://bolinlai.github.io/Lego_EgoActGen/) | [Paper](https://arxiv.org/pdf/2312.03849) | [Dataset](https://www.dropbox.com/scl/fo/4m0v9oy753aimas8rz6v1/ANoJhZQz2BdcGIVLzUsHdP0?rlkey=o8saklcszfc098mjnpid767ic&dl=0)

#### <font color=red>**Our dataset has been released!**</font>

Thank you for your interest in our work! The first version of the code has been released. We are editing the README instructions.

ToDo:

- [x] Dataset

- [x] Codes

- [ ] README (Updating...)

- [ ] Checkpoints

 <img src='https://bolinlai.github.io/Lego_EgoActGen/figures/visualization_new_actions.png'/>



## Contents
- [Setup](#setup)
- [Model Weights](#model-weights)
- Dataset
- [Train](#train)
- [Evaluation](#evaluation)

## Setup

Due to the incompatibility of VLLM and LDM packages, we use two environments for each model.

Install all dependencies with:

```shell
conda env create -f vllm_env.yaml  # set up environment for vllm
conda env create -f ldm_env.yaml  # set up environment for ldm
```

Download the pretrained models by running:
```shell

```