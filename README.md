# diff-sampler
diff-sampler is an open source toolbox for fast sampling of diffusion models, with various model implementations, numerical-based solvers, time schedules and other features. 
This repository also includes (or will include) the official implementations of the following works:

- [ ] [A Geometric Perspective on Diffusion Models](https://arxiv.org/abs/2305.19947)
- [x] [CVPR 2024] [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)
- [ ] [ICML 2024] [On the Trajectory Regularity of ODE-based Diffusion Sampling](https://arxiv.org/abs/2405.11326)

## News
- **2024-06-03**. The repo is made easier to use. Now the pre-trained models will be automatically downloaded to `./src/dataset_name`. Some errors and typos are fixed. Detailed running scripts are provided in `launch.sh`, where we also add new scripts for evaluation of CLIP score for Stable Diffusion.
- **2024-05-02**. Our work [On the Trajectory Regularity of ODE-based Diffusion Sampling](https://arxiv.org/abs/2405.11326) is accepted by ICML 2024.
- **2024-03-25**. The official implementation of the paper [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](./amed-solver-main/) and a [toolbox](./diff-solvers-main/) for fast sampling of diffusion models is released. We upload the reference statistics for FID evaluation [here](https://drive.google.com/drive/folders/1f8qf5qtUewCdDrkExK_Tk5-qC-fNPKpL?usp=sharing). Hope that this repo can facilitate researchers on fast sampling of diffusion models!
- **2024-02-27**. Our work [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094) is accepted by CVPR 2024.

## Supported Fast Samplers for Diffusion Models
| Name | Max Order | Source | Location |
|------|-----------|--------|----------|
|Euler|1|[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)|[diff-solvers-main](./diff-solvers-main/)|
|Heun|2|[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)|[diff-solvers-main](./diff-solvers-main/)|
|DPM-Solver-2|2|[DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)|[diff-solvers-main](./diff-solvers-main/)|
|DPM-Solver++|3|[DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)|[diff-solvers-main](./diff-solvers-main/)|
|UniPC|3|[UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html)|[diff-solvers-main](./diff-solvers-main/)|
|DEIS|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|[diff-solvers-main](./diff-solvers-main/)|
|iPNDM|4|[Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)|[diff-solvers-main](./diff-solvers-main/)|
|iPNDM_v|4|The variable-step version of the Adams–Bashforth methods|[diff-solvers-main](./diff-solvers-main/)|
|AMED-Solver|2|[Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)|[amed-solver-main](./amed-solver-main/)|
|AMED-Plugin|-|[Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)|[amed-solver-main](./amed-solver-main/)|


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
