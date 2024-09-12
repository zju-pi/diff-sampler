import re
import ast
import pickle
import dnnlib
import PIL.Image
import numpy as np
import torch
from torch_utils import distributed as dist
from torch_utils import misc
from torch_utils.download_util import check_file_by_key
from torchvision.utils import make_grid, save_image
import solvers
import solver_utils
from IPython.display import display

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **solver_kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **solver_kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **solver_kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **solver_kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None, accelerator=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    if accelerator is not None:
        accelerator.print(f'Loading the pre-trained diffusion model from "{model_path}"...')
    else:
        dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
        model_source = 'edm'
    elif dataset_name in ['lsun_bedroom', 'lsun_cat']:                      # models from Consistency Models
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = 'cm'
    else:
        if guidance_type == 'cg':            # clssifier guidance           # models from ADM
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
            model_source = 'adm'
        elif guidance_type in ['uncond', 'cfg']:                            # models from LDM
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ffhq_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/ffhq-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
            model_source = 'ldm'
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net, model_source

#----------------------------------------------------------------------------

def fid_prepare(ref_stat_path, inceptionV3_path, device, accelerator=None):
    assert ref_stat_path is not None, 'Specify the path to the reference statistics for FID evaluation, \
    (provided in https://drive.google.com/file/d/196tB1pdpFzZ4cAuHxF_p46P1Aw37bUHz/view?usp=drive_link).'
    if accelerator is not None:
        accelerator.print(f'Loading the reference statistics for FID evaluaiton...')
    else:
        dist.print0(f'Loading the reference statistics for FID evaluaiton...')
    with dnnlib.util.open_url(ref_stat_path) as f:
        ref_stat = dict(np.load(f))
    mu_ref, sigma_ref = ref_stat['mu'], ref_stat['sigma']
    dist.print0(f'Finished.') if accelerator is None else accelerator.print(f'Finished.')

    if accelerator is not None:
        accelerator.print('Loading Inception-v3 model for FID evaluation...')
    else:
        dist.print0('Loading Inception-v3 model for FID evaluation...')
    if inceptionV3_path is not None:
        model = open(inceptionV3_path, 'rb')
        detector_net = pickle.load(model).to(device)
    else:
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
            detector_net = pickle.load(f).to(device)
    dist.print0(f'Finished.') if accelerator is None else accelerator.print(f'Finished.')
    
    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    return mu, sigma, mu_ref, sigma_ref, detector_net

#----------------------------------------------------------------------------

def cifar10_prepare(path_to_cifar10, device, accelerator=None):
    dist.print0('Loading CIFAR-10 dataset...') if accelerator is None else accelerator.print('Loading CIFAR-10 dataset...')
    dataset_kwargs = dnnlib.EasyDict(class_name='dataset.ImageFolderDataset', path=path_to_cifar10, use_labels=False, cache=True, resolution=32, max_size=50000)
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=4, prefetch_factor=2)
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=0, shuffle=False)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=50000, **data_loader_kwargs))

    cifar10_dataset, _ = next(dataset_iterator)
    cifar10_dataset = cifar10_dataset.to(device).to(torch.float32) / 127.5 - 1
    dist.print0(f'Finished.') if accelerator is None else accelerator.print(f'Finished.')

    return cifar10_dataset

#----------------------------------------------------------------------------

def configure_solver(solver_kwargs, net, device, accelerator=None):
    # Get the time schedule
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    if solver_kwargs['t_steps'] is None:
        t_steps = solver_utils.get_schedule(solver_kwargs['num_steps'], solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                            schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], net=net)
    else:
        t_steps_list = ast.literal_eval(solver_kwargs['t_steps'])
        t_steps = torch.tensor(t_steps_list, device=device)
        solver_kwargs['num_steps'] = t_steps.shape[0]
        solver_kwargs['sigma_max'], solver_kwargs['sigma_min'] = t_steps_list[0], t_steps_list[-1]
        solver_kwargs['schedule_type'] = solver_kwargs['schedule_rho'] = None
        dist.print0('Pre-specified t_steps:', t_steps_list) if accelerator is None else accelerator.print('Pre-specified t_steps:', t_steps_list)

    solver_kwargs['t_steps'] = t_steps

    # Calculate the exact NFE
    solver = solver_kwargs['solver']
    if solver in ['dpm', 'heun']:                           # 1 step = 2 NFE
        nfe = 2 * (solver_kwargs['num_steps'] - 1) - 1 if solver_kwargs['afs'] else 2 * (solver_kwargs['num_steps'] - 1)
    else:                                                   # 1 step = 1 NFE
        nfe = solver_kwargs['num_steps'] - 2 if solver_kwargs['afs'] else solver_kwargs['num_steps'] - 1
    if solver_kwargs['denoise_to_zero']:                    # need another 1 NFE, not recommend
        nfe += 1
    if solver_kwargs['dataset_name'] in ['ms_coco, sdxl'] and solver_kwargs['guidance_rate'] not in [0., 1.]:
        nfe = 2 * nfe
    solver_kwargs['nfe'] = nfe

    # Construct solver, 8 solvers are provided
    if solver == 'euler':
        sampler_fn = solvers.euler_sampler
    elif solver == 'heun':
        sampler_fn = solvers.heun_sampler
    elif solver == 'dpm':
        sampler_fn = solvers.dpm_2_sampler
    elif solver == 'ipndm':
        sampler_fn = solvers.ipndm_sampler
    elif solver == 'ipndm_v':
        sampler_fn = solvers.ipndm_v_sampler
    elif solver == 'dpmpp':
        sampler_fn = solvers.dpm_pp_sampler
    elif solver == 'deis':
        sampler_fn = solvers.deis_sampler   # use deis_tab algorithm by default
        # Construct a matrix to store the problematic coefficients for every sampling step
        solver_kwargs['coeff_list'] = solver_utils.get_deis_coeff_list(t_steps, solver_kwargs['max_order'], deis_mode=solver_kwargs["deis_mode"])

    # Print solver settings.
    dist.print0("Solver settings:") if accelerator is None else accelerator.print("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'max_order' and solver in ['euler', 'heun', 'dpm']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']:
            continue
        elif key in ['deis_mode'] and solver not in ['deis']:
            continue
        elif key in ['prompt'] and solver_kwargs['dataset_name'] not in ['ms_coco']:
            continue
        elif key in ['guidance_rate', 'guidance_type'] and solver_kwargs['model_source'] not in ['adm', 'ldm']:
            continue
        elif key in ['coeff_list']:
            continue
        elif key in ['return_inters', 'return_denoised', 'return_eps']:
            continue
        dist.print0(f"\t{key}: {value}") if accelerator is None else accelerator.print(f"\t{key}: {value}")
    
    return sampler_fn, solver_kwargs

#----------------------------------------------------------------------------
# Calculate the deviation of the sampling trajectory

def cal_deviation(traj, ch, r, bs=1):
    traj = traj.transpose(0, 1)
    # intermedia points, start point, end point
    a, b, c = traj[:, 1:-1], traj[:, 0].unsqueeze(1), traj[:, -1].unsqueeze(1)

    ac = c - a                                                                          # (bs, num_steps-1, ch, r, r)
    bc = c - b                                                                          # (bs, 1, ch, r, r)
    bc_unit = bc / torch.norm(bc, p=2, dim=(1, 2, 3, 4)).reshape(bs, 1, 1, 1, 1)        # (bs, 1, ch, r, r)
    
    # Calculate projection vector
    bc_unit_bcasted = bc_unit.expand_as(ac)                                             # (bs, num_steps-1, ch, r, r)
    temp = torch.sum(ac * bc_unit_bcasted, dim=(2, 3, 4))                               # (bs, num_steps-1,)
    temp_expanded = temp.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, ch, r, r)  # (bs, num_steps-1, ch, r, r)
    ac_projection = temp_expanded * bc_unit
    
    # Calculate the deviation
    perp = ac - ac_projection                                                           # (bs, num_steps-1, ch, r, r)
    norm = torch.norm(perp, p=2, dim=(2, 3, 4))
    return norm  

#----------------------------------------------------------------------------

def display_image_grid(images, nrows=None):
    if nrows is None:
        nrows = int(images.shape[0]) // int(images.shape[0] ** 0.5)
    image_grid = make_grid(images, nrows, padding=0)
    images_np = (image_grid * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    images = PIL.Image.fromarray(images_np)
    display(images)

#----------------------------------------------------------------------------
    
def gather(accelerator, tensor):
    if accelerator.num_processes == 1:
        return tensor
    
    return accelerator.gather(tensor.transpose(0,1)).transpose(0,1)