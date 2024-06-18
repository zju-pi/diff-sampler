# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerOutput

class DPMSolverMultistepScheduler(DefaultDPMSolver):
    
    def set_timesteps(
        self,           
        num_inference_steps: int = None, 
        device: Union[str, torch.device] = None,
        timesteps: List[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return

        assert hasattr(self, "scale_dirs"), "scale_dirs must be set before calling set_timesteps"
        assert hasattr(self, "scale_times"), "scale_times must be set before calling set_timesteps"
        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(device=device, dtype=torch.int64) # Ignore the last 0

        # Scale odd-indexed components of self.sigmas by self.scale_times and retrieve timesteps from scaled sigmas
        for i in range(len(self.scale_times)):
            if i % 2 == 1:
                sigma_target = self.sigmas[i] * self.scale_times[i]
                sigmas_source = torch.tensor(all_sigmas[timesteps[i+1]+1:timesteps[i-1]])
                self.timesteps[i] = timesteps[i+1] + 1 + torch.argmin(torch.abs(sigmas_source - sigma_target)).data
            
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None,] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    def dpm_solver_first_order_update(
        self,
        model_output: torch.FloatTensor,
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        scale_dir: Optional[int] = 1,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from the learned diffusion model.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - scale_dir * (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - scale_dir * (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + scale_dir * (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (
                (alpha_t / alpha_s) * sample
                - scale_dir * 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        noise: Optional[torch.FloatTensor] = None,
        scale_dir: Optional[int] = 1,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """
        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - scale_dir * (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - scale_dir * 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - scale_dir * (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + scale_dir * (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - scale_dir * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - scale_dir * 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - scale_dir * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - scale_dir * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + scale_dir * (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + scale_dir * 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + scale_dir * (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + scale_dir * (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - scale_dir * 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - scale_dir * (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - scale_dir * 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - scale_dir * 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.FloatTensor],
        *args,
        sample: torch.FloatTensor = None,
        scale_dir: Optional[int] = 1,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        One step for the third-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by diffusion process.

        Returns:
            `torch.FloatTensor`:
                The sample tensor at the previous timestep.
        """

        timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(" missing`sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0) * sample
                - scale_dir * (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + scale_dir * (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - scale_dir * (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (alpha_t / alpha_s0) * sample
                - scale_dir * (sigma_t * (torch.exp(h) - 1.0)) * D0
                - scale_dir * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - scale_dir * (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        return x_t
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.FloatTensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = randn_tensor(
                model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
            )
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        scale_dir = self.scale_dirs[self.step_index]
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise, scale_dir=scale_dir)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise, scale_dir=scale_dir)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample, scale_dir=scale_dir)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)


