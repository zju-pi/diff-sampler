import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------

@click.command()

# General options.
@click.option('--dataset_name',     help='Dataset name', metavar='STR',                                type=click.Choice(['cifar10', 'ffhq', 'afhqv2', 'imagenet64', 'ms_coco', 'lsun_bedroom_ldm', 'ffhq_ldm']), required=True)
@click.option('--outdir',           help='Where to save the results', metavar='DIR',                   type=str, default="./exps")
@click.option('--total_kimg',       help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--use_step_condition',     help='Weather to use step condition', metavar='BOOL',        type=bool, default=False)
@click.option('--is_second_stage',  help='Second-stage distillation', metavar='BOOL',                  type=bool, default=False)
@click.option('--model_path',       help='Path to pre-trained diffusion model', metavar='DIR',         type=str)

# Options for solvers
@click.option('--num_steps',        help='Number of time steps for si training', metavar='INT',        type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--sampler_tea',      help='Teacher solver', metavar='STR',                              type=click.Choice(['dpm', 'dpmpp', 'euler', 'ipndm', 'heun']), default='dpmpp', show_default=True)
@click.option('--M',                help='Num of steps to interpolate per step', metavar='INT',        type=click.IntRange(min=0), default=3, show_default=True)
@click.option('--guidance_type',    help='Guidance type',                                              type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',    help='Guidance rate', metavar='FLOAT',                             type=float, default=0)
@click.option('--classifier_path',  help='Path to pre-trained classifier model', metavar='DIR',        type=str)
@click.option('--schedule_type',    help='Time discretization schedule', metavar='STR',                type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',     help='Time step exponent', metavar='FLOAT',                        type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--afs',              help='Whether to use afs', metavar='BOOL',                         type=bool, default=True, show_default=True)
# Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, 1<=max_order<=3 for DPM-Solver++
@click.option('--max_order',        help='Order of teacher sampler', metavar='INT',                    type=click.IntRange(min=1), show_default=True)
# Additional options for DPM-Solver++
@click.option('--predict_x0',       help='Whether to use data prediction mode', metavar='BOOL',        type=bool, default=True)
@click.option('--lower_order_final',help='Lower the order at final stages', metavar='BOOL',            type=bool, default=True)

# Hyperparameters.
@click.option('--batch',            help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',               help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=5e-5, show_default=True)      # For cifar10 and ldm distill

# Performance-related.
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)

# I/O-related.
@click.option('--desc',             help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',         help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',             help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--snap',             help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',             help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',             help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('-n', '--dry-run',    help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.loss_kwargs.class_name = 'training.loss.loss'
    c.loss_kwargs.update(num_steps=opts.num_steps, sampler_tea=opts.sampler_tea, \
                         M=opts.m, schedule_type=opts.schedule_type, schedule_rho=opts.schedule_rho, \
                         afs=opts.afs, max_order=opts.max_order, predict_x0=opts.predict_x0, \
                         lower_order_final=opts.lower_order_final, use_step_condition=opts.use_step_condition, \
                         is_second_stage=opts.is_second_stage)

    # Training options.
    c.total_kimg = opts.total_kimg      # Train for total_kimg k trajectories
    c.kimg_per_tick =  1                # total_kimg ticks
    c.snapshot_ticks = c.total_kimg     # 1 snapshots
    c.state_dump_ticks = 99999999       # no dump
    c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
    c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate)
    
    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Description string.
    if opts.schedule_type == 'polynomial':
        schedule_str = 'poly' + str(opts.schedule_rho)
    elif opts.schedule_type == 'logsnr':
        schedule_str = 'logsnr'
    elif opts.schedule_type == 'time_uniform':
        schedule_str = 'uni' + str(opts.schedule_rho)
    elif opts.schedule_type == 'discrete':
        schedule_str = 'discrete' + str(opts.schedule_rho)
    else:
        raise ValueError("Got wrong schedule type: {}".format(opts.schedule_type))
    nfe = opts.num_steps - 2 if opts.afs else opts.num_steps - 1
    nfe = 2 * nfe if opts.dataset_name == 'ms_coco' else nfe
    desc = f'{opts.dataset_name:s}-{opts.num_steps}-{nfe}-{opts.sampler_tea}-{opts.m}-{schedule_str}'
    if opts.afs:
        desc += '-afs'
    if opts.desc:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    c.model_path = opts.model_path
    c.is_second_stage = opts.is_second_stage
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
