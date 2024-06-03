import torch
from torch_utils import persistence
from torch_utils import distributed as dist
import solvers_amed
from solver_utils import get_schedule

#----------------------------------------------------------------------------

def get_solver_fn(solver_name):
    if solver_name == 'amed':
        solver_fn = solvers_amed.amed_sampler
    elif solver_name == 'euler':
        solver_fn = solvers_amed.euler_sampler
    elif solver_name == 'ipndm':
        solver_fn = solvers_amed.ipndm_sampler
    elif solver_name == 'dpm':
        solver_fn = solvers_amed.dpm_2_sampler
    elif solver_name == 'dpmpp':
        solver_fn = solvers_amed.dpm_pp_sampler
    elif solver_name == 'heun':
        solver_fn = solvers_amed.heun_sampler
    else:
        raise ValueError("Got wrong solver name {}".format(solver_name))
    return solver_fn

#----------------------------------------------------------------------------

@persistence.persistent_class
class AMED_loss:
    def __init__(
        self, num_steps=None, sampler_stu=None, sampler_tea=None, M=None, 
        schedule_type=None, schedule_rho=None, afs=False, max_order=None, 
        sigma_min=None, sigma_max=None, predict_x0=True, lower_order_final=True,
    ):
        self.num_steps = num_steps
        self.solver_stu = get_solver_fn(sampler_stu)
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
        
        self.num_steps_teacher = None
        self.tea_slice = None           # a list to extract the intermediate outputs of teacher sampling trajectory
        self.t_steps = None             # baseline time schedule for student
        self.buffer_model = []          # a list to save the history model outputs
        self.buffer_t = []              # a list to save the history time steps

    def __call__(self, AMED_predictor, net, tensor_in, labels=None, step_idx=None, teacher_out=None, condition=None, unconditional_condition=None):
        step_idx = torch.tensor([step_idx]).reshape(1,)
        t_cur = self.t_steps[step_idx].to(tensor_in.device)
        t_next = self.t_steps[step_idx + 1].to(tensor_in.device)
        if step_idx == 0:
            self.buffer_model = []
            self.buffer_t = []

        # Student steps.
        student_out, buffer_model, buffer_t, r, scale_dir, scale_time = self.solver_stu(
            net, 
            tensor_in / t_cur, 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition,
            num_steps=2,
            sigma_min=t_next, 
            sigma_max=t_cur, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=self.afs, 
            denoise_to_zero=False, 
            return_inters=False, 
            AMED_predictor=AMED_predictor, 
            step_idx=step_idx, 
            train=True,
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
            buffer_model=self.buffer_model, 
            buffer_t=self.buffer_t, 
        )
        self.buffer_model = buffer_model
        self.buffer_t = buffer_t
        
        loss = (student_out - teacher_out) ** 2
        dist.print0("Step: {} | Loss: {:8.4f} | r (mean std): {:5.4f} {:5.4f} | scale_dir (mean std): {:5.4f} {:5.4f} | scale_time (mean std): {:5.4f} {:5.4f}".format(
                step_idx.item(), 
                torch.mean(torch.norm(loss, p=2, dim=(1, 2, 3))).item(), 
                r.mean().item(), r.std().item(),
                scale_dir.mean().item(), scale_dir.std().item(),
                scale_time.mean().item(), scale_time.std().item()
            )
        )
        
        return loss, student_out.detach()
    
    def get_teacher_traj(self, net, tensor_in, labels=None, condition=None, unconditional_condition=None):
        if self.t_steps is None:
            self.t_steps = get_schedule(self.num_steps, self.sigma_min, self.sigma_max, schedule_type=self.schedule_type, schedule_rho=self.schedule_rho, device=tensor_in.device, net=net)
        if self.tea_slice is None:
            self.num_steps_teacher = (self.M + 1) * (self.num_steps - 1) + 1
            self.tea_slice = [i * (self.M + 1) for i in range(1, self.num_steps)]
        
        # Teacher steps.
        teacher_traj = self.solver_tea(
            net, 
            tensor_in / self.t_steps[0], 
            class_labels=labels, 
            condition=condition, 
            unconditional_condition=unconditional_condition, 
            num_steps=self.num_steps_teacher, 
            sigma_min=self.sigma_min, 
            sigma_max=self.sigma_max, 
            schedule_type=self.schedule_type, 
            schedule_rho=self.schedule_rho, 
            afs=False, 
            denoise_to_zero=False, 
            return_inters=True, 
            AMED_predictor=None, 
            train=False,
            predict_x0=self.predict_x0, 
            lower_order_final=self.lower_order_final, 
            max_order=self.max_order, 
        )
        
        return teacher_traj[self.tea_slice]
        