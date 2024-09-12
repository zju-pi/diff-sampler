import torch
import numpy as np

#----------------------------------------------------------------------------

def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None, dp_list=None):
    """
    Get the time schedule for sampling.

    Args:
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        device: A torch device.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_type: A `float`. Time step exponent.
        net: A pre-trained diffusion model. Required when schedule_type == 'discrete'.
    Returns:
        a PyTorch tensor with shape [num_steps].
    """
    if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    elif schedule_type == 'logsnr':
        logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
        logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
        t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
        t_steps = (-t_steps).exp()
    elif schedule_type == 'time_uniform':
        epsilon_s = 1e-3
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        step_indices = torch.arange(num_steps, device=device)
        vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
        t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
        t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(t_steps_temp.clone().detach().cpu())
    elif schedule_type == 'discrete':
        assert net is not None
        t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        step_indices = torch.arange(num_steps, device=device)
        t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        t_steps = net.sigma(t_steps_temp)
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))
    
    if dp_list is not None:
        return t_steps[dp_list].to(device)
    return t_steps.to(device)


# Copied from the DPM-Solver codebase (https://github.com/LuChengTHU/dpm-solver).
# Different from the original codebase, we use the VE-SDE formulation for simplicity
# while the official implementation uses the equivalent VP-SDE formulation. 
##############################
### Utils for DPM-Solver++ ###
##############################
#----------------------------------------------------------------------------

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        v: a PyTorch tensor with shape [N].
        dim: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
    
#----------------------------------------------------------------------------

def dynamic_thresholding_fn(x0):
    """
    The dynamic thresholding method
    """
    dims = x0.dim()
    p = 0.995
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, 1. * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

#----------------------------------------------------------------------------

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True):
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1], predict_x0=predict_x0)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

#----------------------------------------------------------------------------

def dpm_solver_first_update(x, s, t, model_s=None, predict_x0=True):
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
    h = lambda_t - lambda_s

    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / s) * x - phi_1 * model_s
    else:
        x_t = x - t * phi_1 * model_s
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-1].reshape(-1, 1, 1, 1)
    lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / t_prev_0) * x - phi_1 * model_prev_0 - 0.5 * phi_1 * D1_0
    else:
        x_t = x - t * phi_1 * model_prev_0 - 0.5 * t * phi_1 * D1_0
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True):
    
    t = t.reshape(-1, 1, 1, 1)
    model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
    
    t_prev_2, t_prev_1, t_prev_0 = t_prev_list[-3], t_prev_list[-2], t_prev_list[-1]
    t_prev_2, t_prev_1, t_prev_0 = t_prev_2.reshape(-1, 1, 1, 1), t_prev_1.reshape(-1, 1, 1, 1), t_prev_0.reshape(-1, 1, 1, 1)
    lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_2.log(), -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_1 = lambda_prev_1 - lambda_prev_2
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0, r1 = h_0 / h, h_1 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
    
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    phi_2 = phi_1 / h + 1. if predict_x0 else phi_1 / h - 1.
    phi_3 = phi_2 / h - 0.5
    # VE-SDE formulation
    if predict_x0:
        x_t = (t / t_prev_0) * x - phi_1 * model_prev_0 + phi_2 * D1 - phi_3 * D2
    else:
        x_t =  x - t * phi_1 * model_prev_0 - t * phi_2 * D1 - t * phi_3 * D2
    return x_t


# A pytorch reimplementation of DEIS (https://github.com/qsh-zh/deis).
#############################
### Utils for DEIS solver ###
#############################
#----------------------------------------------------------------------------
# Transfer from the input time (sigma) used in EDM to that (t) used in DEIS.

def edm2t(edm_steps, epsilon_s=1e-3, sigma_min=0.002, sigma_max=80):
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
    t_steps = vp_sigma_inv(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(edm_steps.clone().detach().cpu())
    return t_steps, vp_beta_min, vp_beta_d + vp_beta_min

#----------------------------------------------------------------------------

def cal_poly(prev_t, j, taus):
    poly = 1
    for k in range(prev_t.shape[0]):
        if k == j:
            continue
        poly *= (taus - prev_t[k]) / (prev_t[j] - prev_t[k])
    return poly

#----------------------------------------------------------------------------
# Transfer from t to alpha_t.

def t2alpha_fn(beta_0, beta_1, t):
    return torch.exp(-0.5 * t ** 2 * (beta_1 - beta_0) - t * beta_0)

#----------------------------------------------------------------------------

def cal_intergrand(beta_0, beta_1, taus):
    with torch.enable_grad():
        taus.requires_grad_(True)
        alpha = t2alpha_fn(beta_0, beta_1, taus)
        log_alpha = alpha.log()
        log_alpha.sum().backward()
        d_log_alpha_dtau = taus.grad
    integrand = -0.5 * d_log_alpha_dtau / torch.sqrt(alpha * (1 - alpha))
    return integrand

#----------------------------------------------------------------------------

def get_deis_coeff_list(t_steps, max_order, N=10000, deis_mode='tab'):
    """
    Get the coefficient list for DEIS sampling.

    Args:
        t_steps: A pytorch tensor. The time steps for sampling.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        N: A `int`. Use how many points to perform the numerical integration when deis_mode=='tab'.
        deis_mode: A `str`. Select between 'tab' and 'rhoab'. Type of DEIS.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """
    if deis_mode == 'tab':
        t_steps, beta_0, beta_1 = edm2t(t_steps)
        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i+1, max_order)
            if order == 1:
                C.append([])
            else:
                taus = torch.linspace(t_cur, t_next, N)   # split the interval for integral appximation
                dtau = (t_next - t_cur) / N
                prev_t = t_steps[[i - k for k in range(order)]]
                coeff_temp = []
                integrand = cal_intergrand(beta_0, beta_1, taus)
                for j in range(order):
                    poly = cal_poly(prev_t, j, taus)
                    coeff_temp.append(torch.sum(integrand * poly) * dtau)
                C.append(coeff_temp)

    elif deis_mode == 'rhoab':
        # Analytical solution, second order
        def get_def_intergral_2(a, b, start, end, c):
            coeff = (end**3 - start**3) / 3 - (end**2 - start**2) * (a + b) / 2 + (end - start) * a * b
            return coeff / ((c - a) * (c - b))
        
        # Analytical solution, third order
        def get_def_intergral_3(a, b, c, start, end, d):
            coeff = (end**4 - start**4) / 4 - (end**3 - start**3) * (a + b + c) / 3 \
                    + (end**2 - start**2) * (a*b + a*c + b*c) / 2 - (end - start) * a * b * c
            return coeff / ((d - a) * (d - b) * (d - c))

        C = []
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            order = min(i, max_order)
            if order == 0:
                C.append([])
            else:
                prev_t = t_steps[[i - k for k in range(order+1)]]
                if order == 1:
                    coeff_cur = ((t_next - prev_t[1])**2 - (t_cur - prev_t[1])**2) / (2 * (t_cur - prev_t[1]))
                    coeff_prev1 = (t_next - t_cur)**2 / (2 * (prev_t[1] - t_cur))
                    coeff_temp = [coeff_cur, coeff_prev1]
                elif order == 2:
                    coeff_cur = get_def_intergral_2(prev_t[1], prev_t[2], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_2(t_cur, prev_t[2], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_2(t_cur, prev_t[1], t_cur, t_next, prev_t[2])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2]
                elif order == 3:
                    coeff_cur = get_def_intergral_3(prev_t[1], prev_t[2], prev_t[3], t_cur, t_next, t_cur)
                    coeff_prev1 = get_def_intergral_3(t_cur, prev_t[2], prev_t[3], t_cur, t_next, prev_t[1])
                    coeff_prev2 = get_def_intergral_3(t_cur, prev_t[1], prev_t[3], t_cur, t_next, prev_t[2])
                    coeff_prev3 = get_def_intergral_3(t_cur, prev_t[1], prev_t[2], t_cur, t_next, prev_t[3])
                    coeff_temp = [coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3]
                C.append(coeff_temp)
    return C

