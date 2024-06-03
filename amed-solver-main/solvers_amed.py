import torch
from solver_utils import *

#----------------------------------------------------------------------------
# Initialize the hook function to get the U-Net bottleneck outputs

def init_hook(net, class_labels=None):
    unet_enc_out = []
    def hook_fn(module, input, output):
        unet_enc_out.append(output.detach())
    if hasattr(net, 'guidance_type'):                                       # models from LDM and Stable Diffusion
        hook = net.model.model.diffusion_model.middle_block.register_forward_hook(hook_fn)
    elif net.img_resolution == 256:                                         # models from CM and ADM with resolution of 256
        hook = net.model.middle_block.register_forward_hook(hook_fn)
    else:                                                                   # models from EDM
        module_name = '8x8_block2' if class_labels is not None else '8x8_block3'
        hook = net.model.enc[module_name].register_forward_hook(hook_fn)
    return unet_enc_out, hook

#----------------------------------------------------------------------------

def get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size):
    if hasattr(net, 'guidance_type') and net.guidance_type == 'classifier-free':
        unet_enc = torch.mean(unet_enc_out[-1], dim=1) if not use_afs else torch.zeros((2*batch_size, 8, 8), device=t_cur.device)
        output = AMED_predictor(unet_enc[batch_size:], t_cur, t_next)
    else:
        unet_enc = torch.mean(unet_enc_out[-1], dim=1) if not use_afs else torch.zeros((batch_size, 8, 8), device=t_cur.device)
        output = AMED_predictor(unet_enc, t_cur, t_next)
    output_list = [*output]
    
    if len(output_list) == 2:
        try:
            use_scale_time = AMED_predictor.module.scale_time
        except:
            use_scale_time = AMED_predictor.scale_time
        if use_scale_time:
            r, scale_time = output_list
            r = r.reshape(-1, 1, 1, 1)
            scale_time = scale_time.reshape(-1, 1, 1, 1)
            scale_dir = torch.ones_like(scale_time)
        else:
            r, scale_dir = output_list
            r = r.reshape(-1, 1, 1, 1)
            scale_dir = scale_dir.reshape(-1, 1, 1, 1)
            scale_time = torch.ones_like(scale_dir)
    elif len(output_list) == 3:
        r, scale_dir, scale_time = output_list
        r = r.reshape(-1, 1, 1, 1)
        scale_dir = scale_dir.reshape(-1, 1, 1, 1)
        scale_time = scale_time.reshape(-1, 1, 1, 1)
    else:
        r = output.reshape(-1, 1, 1, 1)
        scale_dir = torch.ones_like(r)
        scale_time = torch.ones_like(r)
    return r, scale_dir, scale_time

#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):     # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------

def amed_sampler(
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
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    **kwargs
):
    """
    AMED-Solver (https://arxiv.org/abs/2312.00094).

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
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """
    assert AMED_predictor is not None

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        hook.remove()
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * d_mid
    
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

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
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    **kwargs
):  
    """
    AMED-Plugin for Euler sampler.

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
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)

        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
            x_next = x_cur + (t_mid - t_cur) * d_cur
        else:
            x_next = x_cur + (t_next - t_cur) * d_cur
        
        # One more step for student
        if AMED_predictor is not None:
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_mid = (x_next - denoised) / t_mid
            x_next = x_next + scale_dir * (t_next - t_mid) * d_mid
            
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next


#----------------------------------------------------------------------------

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
    AMED_predictor=None, 
    train=False, 
    max_order=4, 
    buffer_model=[], 
    **kwargs
):
    """
    AMED-Plugin for improved PNDM sampler.

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
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        buffer_model: A `list`. History model outputs.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, len(buffer_model)+1)
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
            if order == 1:      # First Euler step.
                x_next = x_cur + (t_mid - t_cur) * d_cur
            elif order == 2:    # Use one history point.
                x_next = x_cur + (t_mid - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # Use two history points.
                x_next = x_cur + (t_mid - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # Use three history points.
                x_next = x_cur + (t_mid - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        else:
            if order == 1:      # First Euler step.
                x_next = x_cur + (t_next - t_cur) * d_cur
            elif order == 2:    # Use one history point.
                x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # Use two history points.
                x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # Use three history points.
                x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
        
        if AMED_predictor is not None:
            order = min(max_order, len(buffer_model)+1)
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_next - denoised) / t_mid
            if order == 1:      # First Euler step.
                x_next = x_next + scale_dir * (t_next - t_mid) * d_cur
            elif order == 2:    # Use one history point.
                x_next = x_next + scale_dir * (t_next - t_mid) * (3 * d_cur - buffer_model[-1]) / 2
            elif order == 3:    # Use two history points.
                x_next = x_next + scale_dir * (t_next - t_mid) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
            elif order == 4:    # Use three history points.
                x_next = x_next + scale_dir * (t_next - t_mid) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
            
            if len(buffer_model) == max_order - 1:
                for k in range(max_order - 2):
                    buffer_model[k] = buffer_model[k+1]
                buffer_model[-1] = d_cur.detach()
            else:
                buffer_model.append(d_cur.detach())
                
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

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
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    r=0.5, 
    **kwargs
):
    """
    AMED-Plugin for DPM-Solver-2.

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
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
        r: A `float`. The hyperparameter controlling the location of the intermediate time step. r=0.5 recovers the original DPM-Solver-2.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            unet_enc_out, hook = init_hook(net, class_labels)
        
        # Euler step.
        use_afs = afs and (((not train) and i == 0) or (train and step_idx == 0))
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        scale_time, scale_dir = 1, 1
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + scale_dir * (t_next - t_cur) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
    
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, [], [], r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

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
    AMED_predictor=None, 
    step_idx=None, 
    train=False, 
    buffer_model=[], 
    buffer_t=[], 
    max_order=3, 
    predict_x0=True, 
    lower_order_final=True,
    **kwargs
):
    """
    AMED-Plugin for multistep DPM-Solver++. 

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
        AMED_predictor: A predictor network.
        step_idx: A `int`. An index to specify the sampling step for training.
        train: A `bool`. In the training loop?
        buffer_model: A `list`. History model outputs.
        buffer_t: A `list`. History time steps.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 3
        predict_x0: A `bool`. Whether to use the data prediction formulation. 
        lower_order_final: A `bool`. Whether to lower the order at the final stages of sampling. 
    Returns:
        A pytorch tensor. The sample at time `sigma_min` or the whole sampling trajectory if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 3
    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    buffer_model = buffer_model if train else []
    buffer_t = buffer_t if train else []
    if AMED_predictor is not None:
        num_steps = 2 * AMED_predictor.module.num_steps - 1 if train else 2 * num_steps - 1
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        if AMED_predictor is not None:
            step_cur = (2 * step_idx + 1 if train else 2 * i + 1)
            unet_enc_out, hook = init_hook(net, class_labels)
        else:
            step_cur = i + 1
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        buffer_model.append(dynamic_thresholding_fn(denoised)) if predict_x0 else buffer_model.append(d_cur)
        if AMED_predictor is not None:
            hook.remove()
            t_cur = t_cur.reshape(-1, 1, 1, 1)
            t_next = t_next.reshape(-1, 1, 1, 1)
            r, scale_dir, scale_time = get_amed_prediction(AMED_predictor, t_cur, t_next, net, unet_enc_out, use_afs, batch_size=latents.shape[0])
            t_mid = (t_next**r) * (t_cur**(1-r))
        buffer_t.append(t_cur)
        
        t_next_temp = t_mid if AMED_predictor is not None else t_next
        if lower_order_final:
            order = step_cur if step_cur < max_order else min(max_order, num_steps - step_cur)
        else:
            order = min(max_order, step_cur)
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_next_temp, order, predict_x0=predict_x0)
            
        # One more step for step instruction:
        if AMED_predictor is not None:
            step_cur = step_cur + 1
            denoised = get_denoised(net, x_next, scale_time * t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            model_out = dynamic_thresholding_fn(denoised) if predict_x0 else ((x_next - denoised) / t_mid)
            buffer_model.append(model_out)
            buffer_t.append(t_mid)
            
            if lower_order_final:
                order = step_cur if step_cur < max_order else min(max_order, num_steps - step_cur)
            else:
                order = min(step_cur, max_order)
            x_next = dpm_pp_update(x_next, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0, scale=scale_dir)
            
        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    if train:
        return x_next, buffer_model, buffer_t, r, scale_dir, scale_time
    return x_next

#----------------------------------------------------------------------------

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
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    # Time step discretization.
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
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
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

