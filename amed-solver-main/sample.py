import os
import re
import csv
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
import solvers_amed

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

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
    elif dataset_name in ['lsun_bedroom']:                                  # models from Consistency Models
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
@click.option('--predictor_path',          help='Path to trained AMED instructor', metavar='DIR',                   type=str, required=True)
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False)

# Options for sampling
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str)
@click.option('--grid',                    help='Whether to make grid',                                             type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         type=bool, default=True, is_flag=True)

def main(predictor_path, max_batch_size, seeds, grid, outdir, subdirs, device=torch.device('cuda'), **solver_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Load models.
    if dist.get_rank() != 0:
        torch.distributed.barrier()     # rank 0 goes first

    # Load AMED predictor
    if not predictor_path.endswith('pkl'):      # load by experiment number
        # find the directory with trained AMED predictor
        predictor_path_str = '0' * (5 - len(predictor_path)) + predictor_path
        for file_name in os.listdir("./exps"):
            if file_name.split('-')[0] == predictor_path_str:
                file_list = [f for f in os.listdir(os.path.join('./exps', file_name)) if f.endswith("pkl")]
                max_index = -1
                max_file = None
                for ckpt_name in file_list:
                    file_index = int(ckpt_name.split("-")[-1].split(".")[0])
                    if file_index > max_index:
                        max_index = file_index
                        max_file = ckpt_name
                predictor_path = os.path.join('./exps', file_name, max_file)
                break
    dist.print0(f'Loading AMED predictor from "{predictor_path}"...')
    with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
        AMED_predictor = pickle.load(f)['model'].to(device)
    
    # Update settings
    prompt = solver_kwargs['prompt']
    solver_kwargs = {key: value for key, value in solver_kwargs.items() if value is not None}
    solver_kwargs['AMED_predictor'] = AMED_predictor
    solver_kwargs['solver'] = solver = AMED_predictor.sampler_stu
    solver_kwargs['num_steps'] = AMED_predictor.num_steps
    solver_kwargs['guidance_type'] = AMED_predictor.guidance_type
    solver_kwargs['guidance_rate'] = AMED_predictor.guidance_rate
    solver_kwargs['afs'] = AMED_predictor.afs
    solver_kwargs['denoise_to_zero'] = False
    solver_kwargs['max_order'] = AMED_predictor.max_order
    solver_kwargs['predict_x0'] = AMED_predictor.predict_x0
    solver_kwargs['lower_order_final'] = AMED_predictor.lower_order_final
    solver_kwargs['schedule_type'] = AMED_predictor.schedule_type
    solver_kwargs['schedule_rho'] = AMED_predictor.schedule_rho
    solver_kwargs['prompt'] = prompt

    solver_kwargs['dataset_name'] = dataset_name = AMED_predictor.dataset_name
    # Load pre-trained diffusion models.
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
    # TODO: support mixed precision 
    # net.use_fp16 = solver_kwargs['use_fp16']

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    # Update settings
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    nfe = 2 * (solver_kwargs['num_steps'] - 1) - 1 if solver_kwargs["afs"] else 2 * (solver_kwargs['num_steps'] - 1)
    nfe = 2 * nfe if dataset_name in ['ms_coco'] else nfe   # should double NFE due to the classifier-free-guidance
    solver_kwargs['nfe'] = nfe

    # Load the prompts
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

    # Construct solver, 5 solvers are provided
    if solver == 'amed':
        sampler_fn = solvers_amed.amed_sampler
    elif solver == 'euler':
        sampler_fn = solvers_amed.euler_sampler
    elif solver == 'dpm':
        sampler_fn = solvers_amed.dpm_2_sampler
    elif solver == 'ipndm':
        sampler_fn = solvers_amed.ipndm_sampler
    elif solver == 'dpmpp':
        sampler_fn = solvers_amed.dpm_pp_sampler
    
    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'AMED_predictor':
            continue
        elif key == 'max_order' and solver in ['euler', 'dpm']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
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
            if solver_kwargs['model_source'] == 'adm':                                              # ADM models
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