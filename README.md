# diff-sampler
diff-sampler is an open source toolbox for fast sampling of diffusion models, with various model implementations, numerical-based solvers, time schedules and other features. 
This repository also includes official implementations of the following works:

- [ ] **A Geometric Perspective on Diffusion Models**, https://arxiv.org/abs/2305.19947
- [x] **Fast ODE-based Sampling for Diffusion Models in Around 5 Steps**, [CVPR 2024], https://arxiv.org/abs/2312.00094
- [ ] **On the Trajectory Regularity of ODE-based Diffusion Sampling**, [ICML 2024], https://arxiv.org/abs/2405.11326

## Requirements
- This repository is mainly built upon [EDM](https://github.com/NVlabs/edm). To install the required packages, please refer to the [EDM](https://github.com/NVlabs/edm) codebase.
- This codebase supports the pre-trained diffusion models from [EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), [Consistency models](https://github.com/openai/consistency_models), [LDM](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Please refer to the corresponding codebases for package installation, if you want to load their pre-trained diffusion models.

## Supported ODE Solvers for Diffusion Models
| Name | Max Order | Source | Location |
|------|-----------|--------|----------|
|Euler|1|[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)|diff-solvers-main|
|Heun|2|[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)|diff-solvers-main|
|DPM-Solver-2|2|[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)|diff-solvers-main|
|AMED-Solver|2|[Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)|amed-solver-main|
|DPM-Solver++|3|[DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)|diff-solvers-main|
|UniPC|3|[UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html)|diff-solvers-main|
|DEIS|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|diff-solvers-main|
|iPNDM|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|diff-solvers-main|
|iPNDM_v|4|The variable-step version of the Adams–Bashforth methods|diff-solvers-main|
|AMED-Plugin|4|[Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)|amed-solver-main|

## Pre-trained Diffusion Models
We perform sampling on a variaty of pre-trained diffusion models from different codebases including
[EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), [Consistency models](https://github.com/openai/consistency_models), [LDM](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion). The tested pre-trained models are listed below:

| Codebase | Dataset | Resolusion | Pre-trained Models | Description |
|----------|---------|------------|--------------------|-------------|
|EDM|CIFAR10|32|[edm-cifar10-32x32-uncond-vp.pkl](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl)
|EDM|FFHQ|64|[edm-ffhq-64x64-uncond-vp.pkl](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl)
|EDM|ImageNet|64|[edm-imagenet-64x64-cond-adm.pkl](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl)
|Consistency Models|LSUN_bedroom|256|[edm_bedroom256_ema.pt](https://openaipublic.blob.core.windows.net/consistency/edm_bedroom256_ema.pt)|Pixel-space
|ADM|ImageNet|256|[256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt) and [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)|Classifier-guidance.
|LDM|LSUN_bedroom|256|[lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt) and [vq-f4 model](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip)|Latent-space
|Stable Diffusion|MS-COCO|512|[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)|Classifier-free-guidance

## FID Statistics
For facilitating the FID evaluation of diffusion models, we provide our [FID statistics](https://drive.google.com/drive/folders/1f8qf5qtUewCdDrkExK_Tk5-qC-fNPKpL?usp=sharing) of various datasets. They are collected on the Internet or made by ourselves with the guidance of the [EDM](https://github.com/NVlabs/edm) codebase. 

## Citation
If you find this repository useful, please consider citing the following paper:

```
@article{zhou2023fast,
  title={Fast ODE-based Sampling for Diffusion Models in Around 5 Steps},
  author={Zhou, Zhenyu and Chen, Defang and Wang, Can and Chen, Chun},
  journal={arXiv preprint arXiv:2312.00094},
  year={2023}
}

@article{chen2024trajectory,
  title={On the Trajectory Regularity of ODE-based Diffusion Sampling},
  author={Chen, Defang and Zhou, Zhenyu and Wang, Can and Shen, Chunhua and Lyu, Siwei},
  journal={arXiv preprint arXiv:2405.11326},
  year={2024}
}
```
