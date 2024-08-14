import numpy as np
import time

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=np.float32,
    ):
        if schedule not in ['discrete', 'linear']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear'".format(schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * np.log(1 - betas).cumsum(axis=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * np.log(alphas_cumprod)
            self.T = 1.
            self.log_alpha_array = self.numerical_clip_alpha(log_alphas).reshape((1, -1,)).astype(dtype)
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = np.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).astype(dtype)
        else:
            self.T = 1.
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues.
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * np.log(1. - np.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = np.searchsorted(np.flip(lambs, axis=0), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array,
                                  self.log_alpha_array).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return np.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return np.sqrt(1. - np.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * np.log(1. - np.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * np.logaddexp(-2. * lamb, np.zeros((1,)))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (np.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * np.logaddexp(np.zeros((1,)), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)),
                               np.flip(self.log_alpha_array, axis=1),
                               np.flip(self.t_array, axis=1))
            return t.reshape((-1,))


def model_wrapper(
        model,
        noise_schedule,
        model_type="noise",
        model_kwargs={},
        guidance_type="uncond",
        condition=None,
        unconditional_condition=None,
        guidance_scale=1.,
        classifier_fn=None,
        classifier_kwargs={},
):
    def get_model_input_time(t_continuous):

        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        # guidance_type == "classifier-free":
        if guidance_scale == 1. or unconditional_condition is None:
            return noise_pred_fn(x, t_continuous, cond=condition)
        else:
            # x_in = torch.cat([x] * 2)
            # t_in = torch.cat([t_continuous] * 2)
            # c_in = torch.cat([unconditional_condition, condition])
            x_in = np.repeat(x, repeats=2, axis=0)  # Repeat x twice and concatenate  np.repeat(x, repeats=2, axis=0)
            t_in = np.repeat(t_continuous, repeats=2, axis=0)  # Repeat t_continuous twice and concatenate
            c_in = np.concatenate([unconditional_condition, condition])  # Concatenate unconditional_condition and condition
            noise_uncond, noise = np.split(noise_pred_fn(x_in, t_in, cond=c_in),2)
            return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_x0_fn=None,
            correcting_xt_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
    ):
        """
        Construct a DPM-Solver.
        """
        # self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.model = lambda x, t: model_fn(x, np.tile(t, x.shape[0]))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N):
        """
        Compute the intermediate time steps for sampling.
        """
        if skip_type == 'time_uniform':
            return np.linspace(t_T, t_0, N + 1)
        
    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = np.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = np.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                    sigma_t / sigma_s * x
                    - alpha_t * phi_1 * model_s
            )
            return x_t
            
    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(
            t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = np.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = np.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = (
                        (sigma_t / sigma_prev_0) * x
                        - (alpha_t * phi_1) * model_prev_0
                        - 0.5 * (alpha_t * phi_1) * D1_0
                )
        return x_t

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
               method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
               atol=0.0078, rtol=0.05, return_intermediate=False,
               ):
        
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        # multi-step method

        assert steps >= order
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps)

        assert timesteps.shape[0] - 1 == steps
        # Init the initial values.
        step = 0
        t = timesteps[step]
        t_prev_list = [t]
        model_prev_list = [self.model_fn(x, t)]

        # Init the first `order` values by lower order multistep DPM-Solver.
        for step in range(1, order):
            t = timesteps[step]
            x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step,
                                                    solver_type=solver_type)

            t_prev_list.append(t)
            model_prev_list.append(self.model_fn(x, t))

        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(order, steps + 1):
            step_start = time.time()

            t = timesteps[step]
            # We only use lower order for steps < 10
            # if lower_order_final and steps < 10:
            if lower_order_final:   # recommended by Shuchen Xue
                step_order = min(order, steps + 1 - step)
            
            x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order,
                                                    solver_type=solver_type)
            
            for i in range(order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]

            t_prev_list[-1] = t

            # We do not need to evaluate the final model value.
            if step < steps:
                model_prev_list[-1] = self.model_fn(x, t)

            step_end = time.time()
            step_duration = step_end - step_start
            # print(f'step {step}/{steps} finished')
            print(f'step total time {step_duration}s')

        return x


#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    # all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    all_x = np.concatenate([x[:, :, np.newaxis], xp[:, np.newaxis, :]], axis=2)
    # sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    # x_idx = torch.argmin(x_indices, dim=2)
    sorted_all_x = np.sort(all_x, axis=2)
    x_indices = np.argsort(all_x, axis=2)
    x_idx = np.argmin(x_indices, axis=2)

    cand_start_idx = x_idx - 1

    # start_idx = torch.where(
    #     torch.eq(x_idx, 0),
    #     torch.tensor(1, device=x.device),
    #     torch.where(
    #         torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
    #     ),
    # )

    start_idx = np.where(
        np.equal(x_idx, 0),
        1,
        np.where(
            np.equal(x_idx, K), K - 2, cand_start_idx,
        ),
    )

    # end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    # start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    # end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    # start_idx2 = torch.where(
    #     torch.eq(x_idx, 0),
    #     torch.tensor(0, device=x.device),
    #     torch.where(
    #         torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
    #     ),
    # )

    end_idx = np.where(np.equal(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = np.take_along_axis(sorted_all_x, start_idx[:, :, np.newaxis], axis=2).squeeze(2)
    end_x = np.take_along_axis(sorted_all_x, end_idx[:, :, np.newaxis], axis=2).squeeze(2)
    start_idx2 = np.where(
        np.equal(x_idx, 0),
        0,
        np.where(
            np.equal(x_idx, K), K - 2, cand_start_idx,
        ),
    )

    # y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    # start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    # end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    # cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)

    y_positions_expanded = np.expand_dims(yp, axis=0).repeat(N, axis=0)
    start_y = np.take_along_axis(y_positions_expanded, start_idx2[:, :, np.newaxis], axis=2).squeeze(2)
    end_y = np.take_along_axis(y_positions_expanded, (start_idx2 + 1)[:, :, np.newaxis], axis=2).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a Numpy Array with shape [N].
        `dim`: a `int`.
    Returns:
        a Numpy Array with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return np.expand_dims(v, axis=tuple(range(1, dims)))