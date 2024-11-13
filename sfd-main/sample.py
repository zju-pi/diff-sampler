import os
import re
import csv
import json
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import solvers
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch import autocast
from torch_utils.download_util import check_file_by_key

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

@click.command()
# General options
@click.option('--model_path',              help='Path to pre-trained diffusion model', metavar='DIR',               type=str, required=True)
@click.option('--dataset_name',            help='Name of the dataset', metavar='STR',                               type=str, required=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--num_steps',               help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=4, show_default=True)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str)
@click.option('--grid',                    help='Whether to make grid',                                             type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         type=bool, default=True, is_flag=True)

def main(seeds, grid, outdir, subdirs, device=torch.device('cuda'), **solver_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (solver_kwargs['max_batch_size'] * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    dataset_name = solver_kwargs['dataset_name']
    if dataset_name in ['ms_coco']:
        # Loading MS-COCO captions
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load distilled models.
    model_path = solver_kwargs['model_path']
    if not model_path.endswith('pkl'):      # load by experiment number
        # find the directory with distilled models
        predictor_path_str = '0' * (5 - len(model_path)) + model_path
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
                model_path = os.path.join('./exps', file_name, max_file)
                break
    dist.print0(f'Loading distilled model from "{model_path}"...')
    with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['model'].to(device)
    net.eval()

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Update settings
    prompt = solver_kwargs['prompt']
    training_kwargs = net.training_kwargs
    assert dataset_name == training_kwargs['dataset_name']
    solver_kwargs = {key: value for key, value in solver_kwargs.items() if value is not None}
    solver_kwargs['guidance_type'] = training_kwargs['guidance_type']
    solver_kwargs['guidance_rate'] = net.guidance_rate = training_kwargs['guidance_rate']
    solver_kwargs['afs'] = training_kwargs['afs']
    solver_kwargs['schedule_type'] = training_kwargs['schedule_type']
    solver_kwargs['schedule_rho'] = training_kwargs['schedule_rho']
    solver_kwargs['model_source'] = training_kwargs['model_source']
    solver_kwargs['sigma_min'] = net.sigma_min = training_kwargs['sigma_min']
    solver_kwargs['sigma_max'] = net.sigma_max = training_kwargs['sigma_max']
    
    solver_kwargs['use_step_condition'] = training_kwargs['use_step_condition']
    solver_kwargs['num_steps'] = solver_kwargs['num_steps'] if solver_kwargs['use_step_condition'] else training_kwargs['num_steps'] # fix num_steps for single distillation
    step_condition = solver_kwargs['num_steps'] if solver_kwargs['use_step_condition'] else None

    nfe = solver_kwargs['num_steps'] - 2 if solver_kwargs["afs"] else solver_kwargs['num_steps'] - 1
    nfe = 2 * nfe if dataset_name in ['ms_coco'] else nfe # should double NFE due to the classifier-free-guidance
    solver_kwargs['nfe'] = nfe

    # Construct solver
    solver = 'euler'
    sampler_fn = solvers.euler_sampler

    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if key == 'max_order' and solver in ['euler', 'dpm', 'heun']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
            continue
        dist.print0(f"\t{key}: {value}")

    # Loop over batches.
    count = 0
    if outdir is None:
        outdir = os.path.join(f"./samples/{dataset_name}", f"{solver}_step{solver_kwargs['num_steps']}_nfe{nfe}")
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        dist.print0('')
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = c = uc = None
        if net.label_dim:
            if solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                if prompt is None:
                    prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                else:
                    prompts = [prompt for i in range(batch_size)]
                uc = None
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
                    images = sampler_fn(net, latents, step_condition=step_condition, condition=c, unconditional_condition=uc, randn_like=rnd.randn_like, **solver_kwargs)
                    images = net.model.decode_first_stage(images)
            else:
                images = sampler_fn(net, latents, step_condition=step_condition, skip_tuning=False, class_labels=class_labels, condition=None, unconditional_condition=None, randn_like=rnd.randn_like, **solver_kwargs)

        # Save images.
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = images.shape[0] // int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))
        else:
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        
        count += batch_size
    
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------