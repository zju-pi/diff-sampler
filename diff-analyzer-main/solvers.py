# Implementation of various ODE solvers for diffusion models.

import torch
from solver_utils import *

#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

@torch.no_grad()
def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------

@torch.no_grad()
def get_denoised_opt(x, t, cifar10_dataset):
    for i in range(x.shape[0]):
        l2_norms = torch.norm(cifar10_dataset - x[i].unsqueeze(0), p=2, dim=(1, 2, 3))  # (50000,)
        weights = torch.nn.functional.softmax((-1 * l2_norms ** 2) / (2 * t ** 2)).reshape(-1, 1, 1, 1)
        if i == 0:
            denoised = torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)
        else:
            denoised = torch.cat((denoised, torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)), dim=0)
    return denoised

#----------------------------------------------------------------------------

@torch.no_grad()
def euler_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    t_steps=None,
    **kwargs
):  
    """
    Euler sampler (equivalent to the DDIM sampler: https://arxiv.org/abs/2010.02502).

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    
    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    
    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    return_denoised=False, 
    return_eps=False, 
    t_steps=None, 
    **kwargs
):
    """
    Heun's second sampler. Introduced in EDM: https://arxiv.org/abs/2206.00364.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def dpm_2_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    r=0.5, 
    t_steps=None,
    **kwargs
):
    """
    DPM-Solver-2 sampler: https://arxiv.org/abs/2206.00927.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        r: A `float`. The hyperparameter controlling the location of the intermediate time step. r=0.5 recovers the original DPM-Solver-2.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_mid
        x_next = x_cur + (t_next - t_cur) * ((1 / (2*r)) * d_prime + (1 - 1 / (2*r)) * d_cur)
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def ipndm_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    max_order=4, 
    t_steps=None,
    **kwargs
):
    """
    Improved PNDM sampler: https://arxiv.org/abs/2204.13902.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    buffer_model = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next

        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
        
    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def ipndm_v_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    max_order=4, 
    t_steps=None,
    **kwargs
):
    """
    The variable-step version of the Adams-Bashforth methods.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    buffer_model = []
    root_d = (latents.shape[1] * latents.shape[-1] ** 2) ** (0.5)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # afs
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def deis_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    max_order=4, 
    coeff_list=None, 
    t_steps=None,
    **kwargs
):
    """
    A pytorch implementation of DEIS: https://arxiv.org/abs/2204.13902.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        coeff_list: A `list`. The pre-calculated coefficients for DEIS sampling.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    assert coeff_list is not None
    
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    buffer_model = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:          # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
            
    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next


#----------------------------------------------------------------------------

@torch.no_grad()
def dpm_pp_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    max_order=3, 
    predict_x0=True, 
    lower_order_final=True, 
    t_steps=None,
    **kwargs
):
    """
    Multistep DPM-Solver++ sampler: https://arxiv.org/abs/2211.01095. 

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 3
        predict_x0: A `bool`. Whether to use the data prediction formulation. 
        lower_order_final: A `bool`. Whether to lower the order at the final stages of sampling. 
    Returns:
        A pytorch tensor. The sample at time `sigma_min` or the whole sampling trajectory if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 3
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    buffer_model = []
    buffer_t = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        buffer_model.append(dynamic_thresholding_fn(denoised)) if predict_x0 else buffer_model.append(d_cur)
        buffer_t.append(t_cur)
        if lower_order_final:
            order = i + 1 if i + 1 < max_order else min(max_order, num_steps - (i + 1))
        else:
            order = min(max_order, i + 1)
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0)
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
       
    if denoise_to_zero:
        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def optimal_sampler(
    net, 
    latents,
    cifar10_dataset, 
    class_labels=None, 
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_denoised=False, 
    return_eps=False, 
    t_steps=None,
    **kwargs
):  
    """
    Optimal Euler sampler (generate images from the dataset).

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters_xt = [x_next.unsqueeze(0)]
    inters_denoised, inters_eps = [], []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            for j in range(x_cur.shape[0]):
                l2_norms = torch.norm(cifar10_dataset - x_cur[j].unsqueeze(0), p=2, dim=(1, 2, 3))  # (50000,)
                weights = torch.nn.functional.softmax((-1 * l2_norms ** 2) / (2 * t_cur ** 2)).reshape(-1, 1, 1, 1)
                if j == 0:
                    denoised = torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)
                else:
                    denoised = torch.cat((denoised, torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)), dim=0)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if return_inters:
            inters_xt.append(x_next.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    
    if denoise_to_zero:
        for j in range(x_cur.shape[0]):
            l2_norms = torch.norm(cifar10_dataset - x_cur[j].unsqueeze(0), p=2, dim=(1, 2, 3))  # (50000,)
            weights = torch.nn.functional.softmax((-1 * l2_norms ** 2) / (2 * t_cur ** 2)).reshape(-1, 1, 1, 1)
            if j == 0:
                denoised = torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)
            else:
                denoised = torch.cat((denoised, torch.sum(torch.mul(cifar10_dataset, weights), dim=0).unsqueeze(0)), dim=0)
        d_cur = (x_next - denoised) / t_next
        if return_inters:
            inters_xt.append(denoised.unsqueeze(0))
        if return_denoised:
            inters_denoised.append(denoised.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    
    if return_inters:
        return torch.cat(inters_xt, dim=0).to(latents.device), torch.cat(inters_denoised, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
    return x_next