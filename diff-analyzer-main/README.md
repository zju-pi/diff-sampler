# diff-analyzer
diff-analyzer is an open source toolbox for analyzing the behavior of the sampling of diffusion models. We offer the code for reproducing some observations in our works:

- [arXiv 2023] [A Geometric Perspective on Diffusion Models](https://arxiv.org/abs/2305.19947)
- [ICML 2024] [On the Trajectory Regularity of ODE-based Diffusion Sampling](https://arxiv.org/abs/2405.11326)

## Requirements
- To install the required packages, please refer to the [EDM](https://github.com/NVlabs/edm) codebase.
- This codebase supports the pre-trained diffusion models from [EDM](https://github.com/NVlabs/edm), [ADM](https://github.com/openai/guided-diffusion), [Consistency models](https://github.com/openai/consistency_models), [LDM](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Please refer to the corresponding codebases for package installation when loading the pre-trained diffusion models from these codebases.

## Getting Started
- ``main.ipynb`` is designed for quick experiments where we only sample several batches for evaluation.
- ``main_mp.ipynb`` is designed for large scale ones where we collect statistics of 50,000 images for more accurate evaluation. It supports **parallel computing** with multiple GPUs using 🤗 Accelerate. The obtained statistics will be saved at ``./outputs``.

## Useful Sources
We provide the required sources (if needed) like processed cifar10 dataset as well as FID statistics [here](https://drive.google.com/drive/folders/1f8qf5qtUewCdDrkExK_Tk5-qC-fNPKpL?).

## Citation
If you find this repository useful, please consider citing the following paper (reverse chronological order):

```bibtex

@article{chen2024trajectory,
  title={On the Trajectory Regularity of ODE-based Diffusion Sampling},
  author={Chen, Defang and Zhou, Zhenyu and Wang, Can and Shen, Chunhua and Lyu, Siwei},
  journal={arXiv preprint arXiv:2405.11326},
  year={2024}
}

@article{chen2023geometric,
  title={A geometric perspective on diffusion models},
  author={Chen, Defang and Zhou, Zhenyu and Mei, Jian-Ping and Shen, Chunhua and Chen, Chun and Wang, Can},
  journal={arXiv preprint arXiv:2305.19947},
  year={2023}
}

```
