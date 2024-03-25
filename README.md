# diff-solvers
diff-solvers is an open source toolbox for fast sampling from diffusion models. We set a strong baseline for fast sampling of diffusion models with ODE-solvers

## Requirements
- This codebase mainly refers to the codebase of [EDM](https://github.com/NVlabs/edm). To install the required packages, please refer to the [EDM](https://github.com/NVlabs/edm) codebase.
- This codebase supports the pre-trained diffusion models from [EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), [Consistency models](https://github.com/openai/consistency_models), [LDM](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion). When you want to load the pre-trained diffusion models from these codebases, please refer to the corresponding codebases for package installation.

## Getting Started
Run the command below to sample with specified ODE solvers and pre-trained diffusion models. This command can be parallelized across multiple GPUs by adjusting ```--nproc_per_node```. You can find the descriptions to all the parameters in the next section. The generated images will be stored at "./samples/dataset_name/" by default.
```.bash
# Generate 50000 images for FID evaluation
SOLVER_FLAGS="--solver=ipndm --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GUIDANCE_FLAGS=""
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="name of the dataset" \
--model_path="/path/to/the/listed/models/above/" \
--batch=64 \
--seeds="0-49999" \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS
```

```.bash
# Generate 16 images in grid form
SOLVER_FLAGS="--solver=ipndm --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=polynomial --schedule_rho=7"
ADDITIONAL_FLAGS="--max_order=4"
GUIDANCE_FLAGS=""
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="name of the dataset" \
--model_path="/path/to/your/model" \
--batch=16 \
--seeds="0-15" \
--grid=True \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS
```

```.bash
# Example: Generate samples on Stable-Diffusion with DPM-Solver++(2M)
SOLVER_FLAGS="--solver=dpmpp --num_steps=6 --afs=False --denoise_to_zero=False"
SCHEDULE_FLAGS="--schedule_type=discrete --schedule_rho=1"
ADDITIONAL_FLAGS="--max_order=2 --predict_x0=False --lower_order_final=True"
GUIDANCE_FLAGS="--guidance_type=cfg --guidance_rate=7.5"
torchrun --standalone --nproc_per_node=1 sample.py \
--dataset_name="ms_coco" \
--model_path="/path/to/stable-diffusion-v1/model.ckpt" \
--batch=4 \
--seeds="0-3" \
--grid=True \
--prompt="a photograph of an astronaut riding a horse" \
$SOLVER_FLAGS \
$SCHEDULE_FLAGS \
$ADDITIONAL_FLAGS \
$GUIDANCE_FLAGS
# --prompt_path="/path/to/MS-COCO_val2014_30k_captions.csv" \  
# add --prompt_path, set --seeds="0-29999", delete --prompt and --grid to generating 30k samples for FID evaluation
```

To compute Fréchet inception distance (FID) for a given model and sampler, first generate 50000 random images and then compare them against the dataset reference statistics using ```fid.py```:
```.bash
# FID evaluation
python fid.py calc --images=path/to/images --ref=path/to/fid/stat
```

You can compute the reference statistics for your own datasets as follows:
```
python fid.py ref --data=path/to/my-dataset.zip --dest=path/to/save/my-dataset.npz
```

## Description of Parameters
| Name | Paramater | Default | Description |
|------|-----------|---------|-------------|
|General options|dataset_name|None|One in ['cifar10', 'ffhq', 'imagenet64', 'lsun_bedroom', 'imagenet256', 'lsun_bedroom_ldm', 'ms_coco']|
|               |model_path|None|Path to the pre-trained diffusion models|
|               |batch|64|Total batch size|
|               |seeds|0-63|Specify a different random seed for each image|
|               |grid|False|Organize the generated images as grid|
|SOLVER_FLAGS|solver|None|One in ['euler', 'heun', 'dpm', 'dpmpp', 'unipc', 'deis', 'ipndm', 'ipndm_v']|
|            |num_steps|6|Number of timestamps. When num_steps=N, there will be N-1 sampling steps. The exact NFE depends on the chosen solver|
|            |afs|False|Whether to use AFS which saves the first model evaluation|
|            |denoise_to_zero|False|Whether to denoise from the last timestamp (>0) to 0. Require one more sampling step|
|SCHEDULE_FLAGS|sigma_min|0.002|Lowest noise level. Specified when loading the pre-trained models|
|              |sigma_max|80.|Highest noise level. Specified when loading the pre-trained models|
|              |schedule_type|'polynomial'|Time discretization schedule. One in ['polynomial', 'logsnr', 'time_uniform', 'discrete']|
|              |schedule_rho|7|Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform', 'discrete']|
|ADDITIONAL_FLAGS|max_order|None|Option for multi-step solvers. 1<=max_order<=4 for iPNDM, iPNDM_v and DEIS, 1<=max_order<=3 for DPM-Solver++ and UniPC|
|                |predict_x0|True|Option for DPM-Solver++ and UniPC. Whether to use the data prediction formulation.|
|                |lower_order_final|True|Option for DPM-Solver++ and UniPC. Whether to lower the order at the final stages of sampling.|
|                |variant|'hb2'|Option for UniPC. One in ['bh1', 'bh2']|
|                |deis_mode|'tab'|Option for UniPC. One in ['tab', 'rhoab']|
|GUIDANCE_FLAGS|guidance_type|None|One in ['cg', 'cfg', 'uncond', None]. 'cg' for classifier-guidance, 'cfg' for classifier-free-guidance used in Stable Diffusion, and 'uncond' for unconditional used in LDM|
|              |guidance_rate|None|Guidance rate|
|              |classifier_path|None|Path to the pre-trained classifier used for classifier-guidance|
|              |prompt|None|Prompt for Stable Diffusion sampling|
|              |prompt_path|None|Path to MS-COCO_val2014_30k_captions.csv for FID evaluation on Stable Diffusion|

## Supported ODE Solvers
| Name | Max Order | Source |
|------|-----------|--------|
|Euler|1|[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)|
|Heun|2|[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)|
|DPM-Solver-2|2|[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)|
|DPM-Solver++|3|[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)|
|UniPC|3|[UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html)|
|DEIS|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|
|iPNDM|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|
|iPNDM_v|4|The variable-step version of the Adams–Bashforth methods|

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

Place the downloaded vq-f4 model at `./models/ldm_models/first_stage_models/vq-f4/model.ckpt`

## FID Statistics
For facilitating the FID evaluation of diffusion models, we provide our [FID statistics](https://drive.google.com/drive/folders/1f8qf5qtUewCdDrkExK_Tk5-qC-fNPKpL?usp=sharing) of various datasets. They are collected on the Internet or made by ourselves with the guidance of the [EDM](https://github.com/NVlabs/edm) codebase. 

## Results
### General Settings
For Euler, Heun, DPM-Solver-2, iPNDM, and iPNDM_v, we use `schedule_type='polynomial'` and `schedule_rho=7` as recommended in the EDM paper (https://arxiv.org/abs/2206.00364).

For DPM-Solver++ and UniPC, we use `schedule_type='logsnr'`, `predict_x0=True` and `lower_order_final=True`. We use `variant='bh2'` for UniPC solver.

For DEIS, we use `schedule_type='time_uniform'`, `schedule_rho=2` and `deis_mode='tab'` for better results.

### FID on CIFAR-10 (unconditional, 32x32)

| Solver | NFE=3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:------:|:-----:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|
|Euler|93.36|66.76|49.66|35.62|27.93|22.32|18.43|15.69|
|Heun|-|319.87|-|99.74|-|38.06|-|15.93|
|DPM-Solver-2|-|145.98|-|60.00|-|10.30|-|5.01|
|DPM-Solver++(3M)|110.03|46.52|24.97|11.99|6.74|4.54|3.42|3.00|
|UniPC-3|109.61|45.20|23.98|11.14|5.83|3.99|3.21|2.89|
|DEIS-tAB3|56.01|25.66|14.39|9.40|6.94|5.55|4.68|4.09|
|iPNDM-4|**47.98**|**24.82**|**13.59**|**7.05**|**5.08**|**3.69**|**3.17**|**2.77**|
|iPNDM_v-4|67.58|40.26|23.58|14.00|9.83|7.34|5.93|4.95|
<!-- |AMED-Solver|18.49|17.18|7.59|7.04|4.36|5.56|3.67|4.14| -->
<!-- |iPNDM + AMED-Plugin|10.81|10.43|6.61|6.67|3.65|3.34|2.63|2.48| -->

### FID on FFHQ (unconditional, 64x64)
| Solver | NFE=3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:------:|:-----:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|
|Euler|78.21|57.48|43.93|35.22|28.86|24.39|21.01|18.37|
|Heun|-|344.87|-|142.39|-|57.21|-|29.54|
|DPM-Solver-2|-|238.57|-|83.17|-|22.84|-|9.46|
|DPM-Solver++(3M)|86.45|45.95|22.51|13.74|8.44|6.04|4.77|4.12|
|UniPC-3|86.43|44.78|21.40|12.85|**7.44**|**5.50**|**4.47**|**3.84**|
|DEIS-tAB3|54.52|28.31|17.36|12.25|9.37|7.59|6.39|5.56|
|iPNDM-4|**45.98**|**28.29**|**17.17**|**10.03**|7.79|5.52|4.58|3.98|
|iPNDM_v-4|60.45|36.80|22.66|15.62|11.57|9.21|7.65|6.55|

### FID on ImageNet (conditional, 64x64)
| Solver | NFE=3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|:------:|:-----:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|
|Euler|82.96|58.43|43.81|34.03|27.46|22.59|19.27|16.72|
|Heun|-|249.41|-|89.63|-|37.65|-|16.46|
|DPM-Solver-2|-|129.75|-|44.83|-|12.42|-|6.84|
|DPM-Solver++(3M)|91.52|56.34|25.49|15.06|10.14|7.84|6.48|5.67|
|UniPC-3|91.38|55.63|24.36|14.30|9.57|7.52|6.34|5.53|
|DEIS-tAB3|**44.51**|**23.53**|**14.75**|**12.57**|**8.20**|**6.84**|5.97|5.34|
|iPNDM-4|58.53|33.79|18.99|12.92|9.17|7.20|**5.91**|**5.11**|
|iPNDM_v-4|65.65|40.20|24.36|16.68|12.23|9.50|7.89|6.76|

## Citation
If you find this repository useful, please consider citing the following paper:

```
@article{zhou2023fast,
  title={Fast ODE-based Sampling for Diffusion Models in Around 5 Steps},
  author={Zhou, Zhenyu and Chen, Defang and Wang, Can and Chen, Chun},
  journal={arXiv preprint arXiv:2312.00094},
  year={2023}
}
```
