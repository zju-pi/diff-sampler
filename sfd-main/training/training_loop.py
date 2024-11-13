"""Main training loop. Fine-tune the whole model"""

import os
import csv
import time
import copy
import json
import pickle
import random
import dnnlib
import numpy as np
import torch
from torch import autocast
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from solver_utils import get_schedule
from models.ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            dist.print0(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    else:
        raise NotImplementedError
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

def create_model(dataset_name=None, model_path=None, guidance_type=None, guidance_rate=None, device=None, is_second_stage=False):
    if is_second_stage: # for second-stage distillation
        assert model_path is not None
        dist.print0(f'Loading the second-stage teacher model from "{model_path}"...')
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['model'].to(device)
        model_source = 'edm' if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64'] else 'ldm'
        return net, model_source

    if model_path is None:
        model_path, _ = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')
    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net_temp = pickle.load(f)['ema'].to(device)
        network_kwargs = dnnlib.EasyDict()
        if dataset_name in ['cifar10']:
            network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
            network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
            network_kwargs.update(dropout=0.13, use_fp16=False)
            network_kwargs.augment_dim = 9
            interface_kwargs = dict(img_resolution=32, img_channels=3, label_dim=0)
        elif dataset_name in ['ffhq', 'afhqv2']:
            network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
            network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2])
            network_kwargs.update(dropout=0.05, use_fp16=False)
            network_kwargs.augment_dim = 9
            interface_kwargs = dict(img_resolution=64, img_channels=3, label_dim=0)
        else:
            network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
            interface_kwargs = dict(img_resolution=64, img_channels=3, label_dim=1000)
        network_kwargs.class_name = 'models.networks_edm.EDMPrecond'
        net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
        net.to(device)
        net.load_state_dict(net_temp.state_dict(), strict=False)
        del net_temp
        net.sigma_min = 0.006
        net.sigma_max = 80.0
        model_source = 'edm'
    elif dataset_name in ['lsun_bedroom_ldm', 'ffhq_ldm', 'ms_coco']:   # models from LDM
        from omegaconf import OmegaConf
        from models.networks_edm import CFGPrecond
        if dataset_name in ['lsun_bedroom_ldm']:
            assert guidance_type == 'uncond'
            config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
            net = load_ldm_model(config, model_path)
            net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            net.sigma_min = 0.006
        elif dataset_name in ['ffhq_ldm']:
            assert guidance_type == 'uncond'
            config = OmegaConf.load('./models/ldm/configs/latent-diffusion/ffhq-ldm-vq-4.yaml')
            net = load_ldm_model(config, model_path)
            net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            net.sigma_min = 0.006
        elif dataset_name in ['ms_coco']:
            assert guidance_type == 'cfg'
            config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
            net = load_ldm_model(config, model_path)
            net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
            net.sigma_min = 0.1
        model_source = 'ldm'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    
    return net, model_source

#----------------------------------------------------------------------------
# Check model structure

def print_network_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module):
            dist.print0(name)

#----------------------------------------------------------------------------

class RandomIntGenerator:
    def __init__(self, seed=42):
        random.seed(seed)

    def randint(self, int_min, int_max):
        while True:
            yield random.randint(int_min, int_max)

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = None,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 20,       # Training duration, measured in thousands of training images.
    kimg_per_tick       = 1,        # Interval of progress prints.
    snapshot_ticks      = 1,        # How often to save network snapshots, None = disable.
    state_dump_ticks    = 99,       # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    dataset_name        = None,
    model_path          = None,
    guidance_type       = None,
    guidance_rate       = 0.,
    device              = torch.device('cuda'),
    is_second_stage     = False,
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_acc_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_acc_rounds * dist.get_world_size()
   
    if dataset_name in ['ms_coco']:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)

    # Load pre-trained diffusion model.
    if dist.get_rank() != 0:
        torch.distributed.barrier()         # rank 0 goes first

    net, model_source = create_model(dataset_name, model_path, guidance_type, guidance_rate, device, is_second_stage)
    if dataset_name in ['ms_coco']:
        net.guidance_rate = 1.0             # training with guidance_rate=1.0, sampling with specified guidance_rate
    net.use_fp16 = True                     # use half precision to accelerate training
    net_copy = copy.deepcopy(net).eval()    # to be the teacher
    net.train().requires_grad_(True)

    if dist.get_rank() == 0:
        torch.distributed.barrier()         # other ranks follow
    
    # Check model structure
    # print_network_layers(net)

    # Check model parameters
    total_params_unet = 0
    for param in net.parameters():
        total_params_unet += param.numel()
    dist.print0("Total parameters in U-Net:     ", total_params_unet)
    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_kwargs.update(sigma_min=net.sigma_min, sigma_max=net.sigma_max, model_source=model_source)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    
    # Record args for sampling
    net.training_kwargs = loss_kwargs
    net.training_kwargs['dataset_name'] = dataset_name
    net.training_kwargs['guidance_type'] = guidance_type
    net.training_kwargs['guidance_rate'] = guidance_rate

    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    rig = RandomIntGenerator()
    num_acc_rounds = 128 // batch_size if dataset_name == 'ms_coco' else 1      # number of accumulation rounds, force 128 for stable diffusion
    batch_gpu_total = num_acc_rounds * batch_gpu
    if guidance_type == 'cfg' and dataset_name in ['ms_coco']:
        with torch.no_grad():
            uc = net.model.get_learned_conditioning(batch_gpu * [""])
    loss = torch.zeros(1,)
    while True:
        if torch.isnan(loss).any().item():
            net.use_fp16 = False 
            net_copy.use_fp16 = False 
            dist.print0('Meet nan, disable fp16!')

        if loss_fn.use_step_condition and not loss_fn.is_second_stage:
            loss_fn.num_steps = next(rig.randint(4, 7))
            loss_fn.M = 2 if loss_fn.num_steps == 3 else 3
            loss_fn.t_steps = get_schedule(loss_fn.num_steps, loss_fn.sigma_min, loss_fn.sigma_max, schedule_type=loss_fn.schedule_type, schedule_rho=loss_fn.schedule_rho, device=device, net=net_copy)
            loss_fn.num_steps_teacher = (loss_fn.M + 1) * (loss_fn.num_steps - 1) + 1
            loss_fn.tea_slice = [i * (loss_fn.M + 1) for i in range(1, loss_fn.num_steps)]

        # Generate latents and conditions in every first step
        latents = [loss_fn.sigma_max * torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device) for k in range(num_acc_rounds)]
        labels = [None for k in range(num_acc_rounds)]
        if net.label_dim:
            if guidance_type == 'cfg' and dataset_name in ['ms_coco']:      # For Stable Diffusion
                prompts = [random.sample(sample_captions, batch_gpu) for k in range(num_acc_rounds)]
                with torch.no_grad():
                    if isinstance(prompts[0], tuple):
                        prompts = [list(p) for p in prompts]
                    c = [net.model.get_learned_conditioning(prompts[k]) for k in range(num_acc_rounds)]
            else:                                                           # EDM models
                labels = [torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_gpu], device=device)] for k in range(num_acc_rounds)]
                
        # Generate teacher trajectories in every first step
        with torch.no_grad():
            if guidance_type in ['uncond', 'cfg']:      # LDM and SD models
                with autocast("cuda"):
                    with net_copy.model.ema_scope():
                        teacher_traj = [loss_fn.get_teacher_traj(net=net_copy, tensor_in=latents[k], labels=labels[k], condition=c[k], unconditional_condition=uc) for k in range(num_acc_rounds)]
            else:
                teacher_traj = [loss_fn.get_teacher_traj(net=net_copy, tensor_in=latents[k], labels=labels[k]) for k in range(num_acc_rounds)]

        # Perform training step by step
        for step_idx in range(loss_fn.num_steps - 1):
            optimizer.zero_grad(set_to_none=True)
            
            # Calculate loss
            for round_idx in range(num_acc_rounds):
                with misc.ddp_sync(ddp, (round_idx == num_acc_rounds - 1)):
                    if guidance_type in ['uncond', 'cfg']:      # LDM and SD models
                        with autocast("cuda"):
                            loss, stu_out = loss_fn(net=ddp, tensor_in=latents[round_idx], labels=labels[round_idx], step_idx=step_idx, teacher_out=teacher_traj[round_idx][step_idx], condition=c[round_idx], unconditional_condition=uc)
                    else:
                        loss, stu_out = loss_fn(net=ddp, tensor_in=latents[round_idx], labels=labels[round_idx], step_idx=step_idx, teacher_out=teacher_traj[round_idx][step_idx])                        
                    latents[round_idx] = stu_out                # start point in next loop
                    training_stats.report('Loss/loss', loss)
                    if not (loss_fn.afs and step_idx == 0):
                        loss.sum().mul(1 / batch_gpu_total).backward()
            
            with torch.no_grad():
                loss_norm = torch.norm(loss, p=2, dim=(1,2,3))
                loss_mean, loss_std = loss_norm.mean().item(), loss_norm.std().item()
            dist.print0("Step: {} | Loss-mean: {:12.8f} | loss-std: {:12.8f}".format(step_idx, loss_mean, loss_std))

            # Update weights.
            if not (loss_fn.afs and step_idx == 0):
                for param in net.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                optimizer.step()

        # Learning rate scheduler
        cur_kimg = cur_nimg / 1000
        if cur_kimg >= 0.5 * total_kimg:
            for g in optimizer.param_groups:
                g['lr'] = optimizer_kwargs['lr'] / 10

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size * num_acc_rounds
        done = (cur_nimg >= total_kimg * 1000)

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        
        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
            
        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and (cur_tick != 0):
            data = dict(model=net)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        # if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            # torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
            # torch.save(dict(net=net), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
