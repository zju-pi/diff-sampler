import ast
import os
import re
import csv
import click
import tqdm
import pickle
import torch
import PIL.Image
import dnnlib
import solvers
import solver_utils
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
from gits_utils import get_dp_list

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

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
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

@click.command()
# General options
@click.option('--dataset_name',            help='Name of the dataset', metavar='STR',                               type=str, required=True)
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)

# Options for sampling
@click.option('--solver',                  help='Name of the solver', metavar='many solvers',                       type=click.Choice(['euler', 'ipndm', 'ipndm_v', 'heun', 'dpm', 'dpmpp', 'deis', 'unipc']), default='ipndm', show_default=True)
@click.option('--num_steps',               help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=7, show_default=True)
@click.option('--afs',                     help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--guidance_type',           help='Guidance type',                                                    type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',           help='Guidance rate',                                                    type=float)
@click.option('--denoise_to_zero',         help='Whether to denoise from the last time step to 0',                  type=bool, default=False)
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)
# Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, iPNDM_v and DEIS, 1<=max_order<=3 for DPM-Solver++ and UniPC
@click.option('--max_order',               help='Max order for solvers', metavar='INT',                             type=click.IntRange(min=1), default=4)
# Additional options for DPM-Solver++ and UniPC
@click.option('--predict_x0',              help='Whether to use data prediction mode', metavar='BOOL',              type=bool, default=True)
@click.option('--lower_order_final',       help='Whether to lower the order at final stages', metavar='BOOL',       type=bool, default=True)
# Additional options for UniPC
@click.option('--variant',                 help='Type of UniPC solver', metavar='STR',                              type=click.Choice(['bh1', 'bh2']), default='bh2')
# Additional options for DEIS
@click.option('--deis_mode',               help='Type of DEIS solver', metavar='STR',                               type=click.Choice(['tab', 'rhoab']), default='tab')

# Options for scheduling
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=0.002)
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=80.)
@click.option('--schedule_type',           help='Time discretization schedule', metavar='STR',                      type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',            help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--t_steps',                 help='Pre-specified time schedule', metavar='STR',                       type=str, default=None)

# DP options.
@click.option('--dp',                      help='Whether to use dp to search a schedule', metavar='BOOL',           type=bool, default=False, show_default=True)
@click.option('--metric',                  help='Metric for cost matrix', metavar='STR',                            type=click.Choice(['dev', 'l1', 'l2']), default='dev', show_default=True)
@click.option('--coeff',                   help='Coefficient for the DP algorithm', metavar='FLOAT',                type=click.FloatRange(min=0, min_open=True), default=1.15, show_default=True)
@click.option('--num_warmup',              help='How many warmup samples for dp', metavar='INT',                    type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--solver_tea',              help='Teacher solver', metavar='STR',                                    type=click.Choice(['euler', 'ipndm', 'ipndm_v', 'heun', 'dpm', 'dpmpp', 'deis']), default='ipndm', show_default=True)
@click.option('--num_steps_tea',           help='Number of timestamps for teacher', metavar='INT',                  type=click.IntRange(min=1), default=21, show_default=True)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str)
@click.option('--grid',                    help='Whether to make grid',                                             type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         type=bool, default=True, is_flag=True)

def main(seeds, grid, outdir, subdirs, t_steps, device=torch.device('cuda'), **solver_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (solver_kwargs['max_batch_size'] * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    dataset_name = solver_kwargs['dataset_name']
    if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)

    # Rank 0 goes first
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load pre-trained diffusion models.
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Get the time schedule
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    if t_steps is None:
        dp_list = get_dp_list(net, device, **solver_kwargs) if solver_kwargs['dp'] else None
        num_steps_in = solver_kwargs['num_steps'] if dp_list is None else solver_kwargs['num_steps_tea']
        t_steps = solver_utils.get_schedule(num_steps_in, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                            schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], \
                                            net=net, dp_list=dp_list)
        if solver_kwargs['dp']:
            dist.print0('Selected dp_list:', dp_list)
            dist.print0('Selected time schedule: ', [round(num.item(), 4) for num in t_steps])
    else:
        if solver_kwargs['dp']:
            dist.print0('t_steps is specified, ignored DP')
        t_steps_list = ast.literal_eval(t_steps)
        t_steps = torch.tensor(t_steps_list, device=device)
        solver_kwargs['num_steps'] = t_steps.shape[0]
        solver_kwargs['sigma_max'], solver_kwargs['sigma_min'] = t_steps_list[0], t_steps_list[-1]
        solver_kwargs['schedule_type'] = solver_kwargs['schedule_rho'] = None
        solver_kwargs['dp'] = False
        dist.print0('Pre-specified t_steps:', t_steps_list)
    solver_kwargs['t_steps'] = t_steps

    # Calculate the exact NFE
    solver = solver_kwargs['solver']
    if solver in ['dpm', 'heun']:                           # 1 step = 2 NFE
        nfe = 2 * (solver_kwargs['num_steps'] - 1)
        if solver_kwargs['afs']:
            # The use of AFS in GITS is to insert a new "free" step in the time schedule, but not totally free for these two methods
            nfe = nfe + 1 if solver_kwargs['dp'] else nfe - 1
    else:                                                   # 1 step = 1 NFE
        nfe = solver_kwargs['num_steps'] - 1
        if solver_kwargs['afs']:
            # The use of AFS in GITS is to insert a new "free" step in the time schedule
            nfe = nfe if solver_kwargs['dp'] else nfe - 1
    if solver_kwargs['denoise_to_zero']:                    # need another 1 NFE, not recommend
        nfe += 1
    if dataset_name in ['ms_coco'] and solver_kwargs['guidance_rate'] not in [0., 1.]:
        # requires doubled NFE due to the classifier-free-guidance
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
    elif solver == 'unipc':
        sampler_fn = solvers.unipc_sampler
    elif solver == 'deis':
        sampler_fn = solvers.deis_sampler   # use deis_tab algorithm by default
        # Construct a matrix to store the problematic coefficients for every sampling step
        solver_kwargs['coeff_list'] = solver_utils.get_deis_coeff_list(t_steps, solver_kwargs['max_order'], deis_mode=solver_kwargs["deis_mode"])

    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'max_order' and solver in ['euler', 'heun', 'dpm']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp', 'unipc']:
            continue
        elif key in ['variant'] and solver not in ['unipc']:
            continue
        elif key in ['deis_mode'] and solver not in ['deis']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
            continue
        elif key in ['t_steps', 'coeff_list']:
            continue
        elif key in ['dp', 'metric', 'coeff', 'num_warmup', 'num_steps_tea', 'solver_tea'] and solver_kwargs['dp'] is False:
            continue
        dist.print0(f"\t{key}: {value}")

    # Loop over batches.
    if outdir is None:
        if grid:
            outdir = os.path.join(f"./samples/grids/{dataset_name}", f"{solver}_nfe{nfe}")
        else:
            outdir = os.path.join(f"./samples/{dataset_name}", f"{solver}_nfe{nfe}")
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = c = uc = None
        if net.label_dim:
            if solver_kwargs['model_source'] == 'adm':
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                if solver_kwargs['prompt'] is None:
                    prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                else:
                    prompts = [solver_kwargs['prompt'] for i in range(batch_size)]
                if solver_kwargs['guidance_rate'] != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
            else:
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

        # Generate images.
        with torch.no_grad():
            if solver_kwargs['model_source'] == 'ldm':
                with autocast("cuda"):
                    with net.model.ema_scope():
                        images = sampler_fn(net, latents, condition=c, unconditional_condition=uc, **solver_kwargs)
                        images = net.model.decode_first_stage(images)
            else:
                images = sampler_fn(net, latents, class_labels=class_labels, **solver_kwargs)

        # Save images.
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))
        else:
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
    
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
