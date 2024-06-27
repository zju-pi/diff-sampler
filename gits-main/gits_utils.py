import csv
import copy
import random
import torch
from torch_utils import distributed as dist
import numpy as np
import solvers
import solver_utils
from torch import autocast
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------
# Get the sampler function

def get_sampler_fn(solver, device, dp_list=None, net=None, **kwargs):
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
        t_steps = solver_utils.get_schedule(kwargs['num_steps_tea'], kwargs['sigma_min'], kwargs['sigma_max'], device=device, \
                                            schedule_type=kwargs["schedule_type"], schedule_rho=kwargs["schedule_rho"], net=net, dp_list=dp_list)
        coeff_list = solver_utils.get_deis_coeff_list(t_steps, kwargs['max_order'], deis_mode=kwargs["deis_mode"])
        return sampler_fn, coeff_list
    else:
        raise NotImplementedError(f"Unknown solver: {solver}")
    return sampler_fn, None

#----------------------------------------------------------------------------
# dp_list is a list of indices to be selected from the longer teacher time schedule

def get_dp_list(net, device, **solver_kwargs):
    kwargs = copy.deepcopy(solver_kwargs)
    dataset_name = kwargs['dataset_name']
    num_warmup = kwargs['num_warmup']
    max_batch_size = kwargs['max_batch_size']
    sigma_min = kwargs['sigma_min']
    sigma_max = kwargs['sigma_max']
    num_steps = kwargs['num_steps']
    num_steps_tea = kwargs['num_steps_tea']
    schedule_type = kwargs['schedule_type']
    schedule_rho = kwargs['schedule_rho']
    afs = kwargs['afs']
    metric = kwargs['metric']
    coeff = kwargs['coeff']
    model_source = kwargs['model_source']

    kwargs['solver'] = solver_kwargs['solver_tea']
    sampler_fn_tea, coeff_list = get_sampler_fn(device=device, net=net, dp_list=[i for i in range(kwargs['num_steps_tea'])], **kwargs)
    kwargs['t_steps'] = t_steps = solvers.get_schedule(num_steps_tea, sigma_min, sigma_max, device=device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    kwargs['coeff_list'] = coeff_list

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
    
    # Calculate the cost matrix
    kwargs['return_inters'] = True
    kwargs['return_eps'] = True
    kwargs['num_steps'] = solver_kwargs['num_steps_tea']
    num_accumulation_rounds = num_warmup // (max_batch_size + 1) + 1
    batch_gpu = max_batch_size // dist.get_world_size()
    dist.print0(f'Accumulate {num_accumulation_rounds} rounds to collect {num_warmup} trajectories...')
    cost_mat = torch.zeros((num_steps_tea, num_steps_tea), device=device)
    for r in range(num_accumulation_rounds):
        with torch.no_grad():
            # Generate latents and labels
            latents = torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            class_labels = c = uc = None
            if net.label_dim:
                if model_source == 'adm':
                    class_labels = torch.randint(net.label_dim, size=(batch_gpu,), device=device)
                elif model_source == 'ldm' and dataset_name == 'ms_coco':
                    if solver_kwargs['prompt'] is None:
                        prompts = random.sample(sample_captions, batch_gpu)
                    else:
                        prompts = [solver_kwargs['prompt'] for i in range(batch_gpu)]
                    if solver_kwargs['guidance_rate'] != 1.0:
                        uc = net.model.get_learned_conditioning(batch_gpu * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = net.model.get_learned_conditioning(prompts)
                else:
                    class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_gpu], device=device)]
            
            dist.print0(f'Round {r+1}/{num_accumulation_rounds} | Generating the teacher trajectory...')
            with torch.no_grad():
                if model_source == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            teacher_traj, eps_traj = sampler_fn_tea(net, latents, condition=c, unconditional_condition=uc, **kwargs)
                else:
                    teacher_traj, eps_traj = sampler_fn_tea(net, latents, class_labels=class_labels, **kwargs)
            dev_tea = cal_deviation(teacher_traj, net.img_channels, net.img_resolution, bs=batch_gpu).mean(dim=0)
            dev_tea = torch.cat([dev_tea, torch.zeros_like(dev_tea[:1])])
            
            dist.print0(f'Round {r+1}/{num_accumulation_rounds} | Calculating the cost matrix...')
            for i in range(num_steps_tea - 1):
                x_cur = teacher_traj[i]
                t_cur = t_steps[i]
                d_cur = eps_traj[i]

                for j in range(i+1, num_steps_tea):
                    t_next = t_steps[j]
                    x_next = x_cur + (t_next - t_cur) * d_cur
                    if metric == 'l1':
                        cost_mat[i][j] += torch.norm(x_next - teacher_traj[j], p=1, dim=(1,2,3)).mean()
                    elif metric == 'l2':
                        cost_mat[i][j] += torch.norm(x_next - teacher_traj[j], p=2, dim=(1,2,3)).mean()
                    elif metric == 'dev':
                        temp = torch.cat((teacher_traj[0].unsqueeze(0), x_next.unsqueeze(0), teacher_traj[-1].unsqueeze(0)), dim=0)
                        dev_stu = cal_deviation(temp, net.img_channels, net.img_resolution, bs=batch_gpu).mean(dim=0)
                        cost_mat[i][j] += (dev_stu - dev_tea[j - 1]).mean()
                    else:
                        raise NotImplementedError(f"Unknown metric: {metric}")

    torch.distributed.all_reduce(cost_mat)
    cost_mat /= dist.get_world_size() * num_accumulation_rounds
    cost_mat = cost_mat.detach().cpu().numpy()

    # Description string.
    if schedule_type == 'polynomial':
        schedule_str = 'poly' + str(schedule_rho)
    elif schedule_type == 'logsnr':
        schedule_str = 'logsnr'
    elif schedule_type == 'time_uniform':
        schedule_str = 'uni' + str(schedule_rho)
    elif schedule_type == 'discrete':
        schedule_str = 'discrete'
    desc = f"{dataset_name}-{solver_kwargs['solver_tea']}-{schedule_str}-{num_steps_tea}-warmup{num_warmup}-{metric}"

    # dynamic programming
    multiple_coeff = True if dataset_name == 'ms_coco' else False
    dp_list = phi = dp(cost_mat, num_steps, num_steps_tea, coeff, multiple_coeff, desc, t_steps)
    
    kwargs['return_inters'] = False
    kwargs['return_eps'] = False
    kwargs['solver'] = solver_kwargs['solver']
    kwargs['num_steps'] = solver_kwargs['num_steps']
    if afs:
        dist.print0('Selecting the AFS step...')
        dist_min = 999999
        for k in range(1, phi[1]):
            dp_slice_temp = copy.deepcopy(phi)
            dp_slice_temp.insert(1, k)
            sampler_fn, solver_kwargs['coeff_list'] = get_sampler_fn(device=device, dp_list=dp_slice_temp, **kwargs)
            kwargs['t_steps'] = solvers.get_schedule(num_steps_tea, sigma_min, sigma_max, device=device, schedule_type=schedule_type, \
                                                     schedule_rho=schedule_rho, net=net, dp_list=dp_slice_temp)
            with torch.no_grad():
                if model_source == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            images_afs = sampler_fn(net, latents, condition=c, unconditional_condition=uc, **kwargs)
                else:
                    images_afs = sampler_fn(net, latents, class_labels=class_labels, **kwargs)
            dist_temp = torch.norm(images_afs - teacher_traj[-1], p=2, dim=(1,2,3)).mean()
            torch.distributed.all_reduce(dist_temp)
            dist_temp /= dist.get_world_size()
            if dist_temp < dist_min:
                dist_min = dist_temp
                dp_list = dp_slice_temp

    return dp_list

#----------------------------------------------------------------------------
# Dynamic programming

def dp(cost_mat, num_steps, num_steps_tea, coeff, multiple_coeff=False, desc=None, t_steps=None):
    K = num_steps - 1
    V = np.full((num_steps_tea, K+1), np.inf)
    for i in range(num_steps_tea):
        V[i][1] = cost_mat[i][-1]
    for k in range(2, K+1):
        for j in range(num_steps_tea - 1):
            for i in range(j + 1, num_steps_tea - 1):
                V[j][k] = min(V[j][k], cost_mat[j][i] + coeff * V[i][k-1])
    phi, w = [0], 0
    for temp in range(K):
        k = K - temp
        for j in range(w + 1, num_steps_tea):
            if V[w][k] == cost_mat[w][j] + coeff * V[j][k-1]:
                phi.append(j)
                w = j
                break
    phi.append(num_steps_tea - 1)
    dp_list = phi

    if multiple_coeff:
        # Output multiple dp_list and time schedule to a txt file with a list of coeffs for efficiency
        K = num_steps - 1
        for coeff in [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.10, 1.15, 1.2]:
            V = np.full((num_steps_tea, K+1), np.inf)
            for i in range(num_steps_tea):
                V[i][1] = cost_mat[i][-1]
            for k in range(2, K+1):
                for j in range(num_steps_tea - 1):
                    for i in range(j + 1, num_steps_tea - 1):
                        V[j][k] = min(V[j][k], cost_mat[j][i] + coeff * V[i][k-1])
            
            if dist.get_rank() == 0:
                Note = open('dp_record.txt', mode='a')
                Note.write(f"{desc}-{coeff}\n")
                for K_temp in range(2, K+1):
                    phi, w = [0], 0
                    for temp in range(K_temp):
                        k = K_temp - temp
                        for j in range(w + 1, num_steps_tea):
                            if V[w][k] == cost_mat[w][j] + coeff * V[j][k-1]:
                                phi.append(j)
                                w = j
                                break
                    phi.append(num_steps_tea - 1)
                    Note.write(f"{phi} {[round(num.item(), 4) for num in t_steps[phi]]}\n")
                Note.close()
    return dp_list

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
