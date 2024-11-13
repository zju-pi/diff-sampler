import torch
from torch_utils import persistence
from torch_utils import distributed as dist
import solvers
from solver_utils import get_schedule
from piq import LPIPS

#----------------------------------------------------------------------------

def get_solver_fn(solver_name):
    if solver_name == 'euler':
        solver_fn = solvers.euler_sampler
    elif solver_name == 'heun':
        solver_fn = solvers.heun_sampler
    elif solver_name == 'dpm':
        solver_fn = solvers.dpm_2_sampler
    elif solver_name == 'ipndm':
        solver_fn = solvers.ipndm_sampler
    elif solver_name == 'dpmpp':
        solver_fn = solvers.dpm_pp_sampler
    else:
        raise ValueError("Got wrong solver name {}".format(solver_name))
    return solver_fn
    
#----------------------------------------------------------------------------

@persistence.persistent_class
class loss:
    def __init__(
        self, num_steps=None, sampler_tea=None, M=None,
        schedule_type=None, schedule_rho=None, afs=False, max_order=None, 
        sigma_min=None, sigma_max=None, predict_x0=True, lower_order_final=True, 
        use_step_condition=False, model_source=None, is_second_stage=False,
    ):
        self.num_steps = num_steps
        self.solver_stu = get_solver_fn('euler')
        self.solver_tea = get_solver_fn(sampler_tea)
        self.M = M
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.use_step_condition = use_step_condition
        self.model_source = model_source
        self.is_second_stage = is_second_stage

        self.num_steps_teacher = None
        self.tea_slice = None           # a list to extract the intermediate outputs of teacher sampling trajectory
        self.t_steps = None             # baseline time schedule for student

        self.lpips = None

    def __call__(self, net, tensor_in, labels=None, step_idx=None, teacher_out=None, condition=None, unconditional_condition=None):
        step_idx = torch.tensor([step_idx]).reshape(1,)
        t_cur = self.t_steps[step_idx].to(tensor_in.device)
        t_next = self.t_steps[step_idx+1].to(tensor_in.device)

        # Student steps.
        student_out, _, _ = self.solver_stu(
            net, 
            tensor_in / t_cur, 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition,
            randn_like=torch.randn_like, 
            num_steps=2,
            sigma_min=t_next, 
            sigma_max=t_cur, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=self.afs, 
            denoise_to_zero=False, 
            return_inters=False, 
            step_idx=step_idx, 
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
            train=True,
            step_condition=self.num_steps if self.use_step_condition else None,
        )

        loss = (student_out - teacher_out).abs()
        if self.is_second_stage and self.model_source == 'edm' and step_idx == self.num_steps - 2: # the last step
            loss += self.get_lpips_measure(student_out, teacher_out).mean()
        
        return loss, student_out.detach()
    
    def get_teacher_traj(self, net, tensor_in, labels=None, condition=None, unconditional_condition=None):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, device=tensor_in.device, net=net)
        if self.tea_slice is None:
            self.num_steps_teacher = (self.M + 1) * (self.num_steps - 1) + 1
            self.tea_slice = [i * (self.M + 1) for i in range(1, self.num_steps)]
        if self.is_second_stage:
            self.solver_tea = solvers.euler_sampler
            self.use_step_condition = net.training_kwargs['use_step_condition']
            self.schedule_type = net.training_kwargs['schedule_type']
            self.schedule_rho = net.training_kwargs['schedule_rho']

        # Teacher steps.
        teacher_traj = self.solver_tea(
            net, 
            tensor_in / self.t_steps[0], 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition, 
            randn_like=torch.randn_like, 
            num_steps=self.num_steps_teacher, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=False if not self.is_second_stage else net.training_kwargs['afs'], 
            denoise_to_zero=False, 
            return_inters=True, 
            return_eps=False, 
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
            train=False,
            step_condition=None if not self.is_second_stage else (self.num_steps_teacher if self.use_step_condition else None), 
        )

        return teacher_traj[self.tea_slice]
    
    def get_lpips_measure(self, img_batch1, img_batch2):
        if self.lpips is None:
            self.lpips = LPIPS(replace_pooling=True, reduction="none")
        out_1 = torch.nn.functional.interpolate(img_batch1, size=224, mode="bilinear")
        out_2 = torch.nn.functional.interpolate(img_batch2, size=224, mode="bilinear")
        return self.lpips(out_1, out_2)
    