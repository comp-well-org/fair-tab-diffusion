"""Modules for denoising diffusion probabilistic models based on Gaussian and multinomial kernels.

References:
    - https://github.com/yandex-research/tab-ddpm/blob/main/tab_ddpm/gaussian_multinomial_diffsuion.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from .utils import log_add_exp, log_1_min_a, extract, mean_flat, normal_kl, discretized_gaussian_log_likelihood
from .utils import sliced_logsumexp, index_to_log_onehot, sum_except_batch, log_categorical, ohe_to_categories

def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar: callable, max_beta: float = 0.2) -> np.ndarray:
    """Create a beta schedule that discretizes the given alpha_t_bar function.
    
    Defines the cumulative product of (1 - beta) over time from t = [0, 1].

    Args:
        num_diffusion_timesteps: number of diffusion steps
        alpha_bar: function that returns alpha_bar(t)
        max_beta: maximum beta value
    
    Returns:
        beta schedule of shape `(num_diffusion_timesteps,)`
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int, max_beta: float = 0.2) -> np.ndarray:
    """Get a pre-defined beta schedule for the given name.
    
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.Beta schedules may be added, but should not 
    be removed or changed once they are committed to maintain backwards compatibility.
    
    Raises:
        NotImplementedError: if schedule_name is not supported

    Args:
        schedule_name: name of the beta schedule
        num_diffusion_timesteps: number of diffusion steps
        max_beta: maximum beta value
    
    Returns:
        beta schedule of shape `(num_diffusion_timesteps,)`
    """
    if schedule_name == 'linear':
        # linear schedule from Ho et al, extended to work for any number of diffusion steps.
        start = 1e-4
        end = 0.02
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * start
        beta_end = scale * end
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64,
        )
    elif schedule_name == 'cosine':
        offset = 0.008
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + offset) / (1 + offset) * math.pi / 2) ** 2,
            max_beta=max_beta,
        )
    raise NotImplementedError(f'unknown beta schedule: {schedule_name}')

class GaussianMultinomialDiffusion(nn.Module):    
    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn: callable,
        num_timesteps: int = 1000,
        gaussian_loss_type: str = 'mse',
        gaussian_parametrization: str = 'eps',  
        multinomial_loss_type: str = 'vb_stochastic',
        parametrization: str = 'x0',
        scheduler: str = 'cosine',
        max_beta: float = 0.2,
        is_fair: bool = True,
        device: str = 'cpu',
    ):
        super().__init__()
        assert gaussian_parametrization in {'eps', 'x0'}
        assert multinomial_loss_type in ('vb_stochastic')
        assert parametrization in {'x0', 'direct'}

        # device
        device = torch.device(device)
        
        # fairness
        self.is_fair = is_fair
        
        # data
        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))]),
        ).to(device)

        # classes
        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        # diffusion
        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        # intermediate variables
        alphas = 1. - get_named_beta_schedule(scheduler, num_timesteps, max_beta=max_beta)
        alphas = torch.tensor(alphas.astype('float64'))
        betas = 1. - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # gaussian diffusion
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.from_numpy(
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])),
        ).float().to(device)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).float().to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas.numpy()) / (1.0 - alphas_cumprod)
        ).float().to(device)

        # check that the diffusion is stable
        threshold = 1e-5
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < threshold
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < threshold
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < threshold

        # convert to float32 and register buffers.
        self.register_buffer('alphas', alphas.float().to(device))
        self.register_buffer('log_alpha', log_alpha.float().to(device))
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float().to(device))
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float().to(device))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float().to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.float().to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float().to(device))
        self.register_buffer('alphas_cumprod_next', alphas_cumprod_next.float().to(device))
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod.float().to(device))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod.float().to(device))
        self.register_buffer('lt_history', torch.zeros(num_timesteps))
        self.register_buffer('lt_count', torch.zeros(num_timesteps))
        
    # gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape,
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        answer = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        answer += extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return answer
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape        
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape,
        )
        b1 = posterior_mean.shape[0]
        b2 = posterior_variance.shape[0]
        b3 = posterior_log_variance_clipped.shape[0]
        b4 = x_start.shape[0]
        assert b1 == b2 == b3 == b4
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def gaussian_p_mean_variance(self, model_output, x_t, t):
        b, _ = x_t.shape[:2]
        assert t.shape == (b,)

        model_variance = torch.cat([self.posterior_variance[1].unsqueeze(0).to(x_t.device), (1. - self.alphas)[1:]], dim=0)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x_t.shape)
        model_log_variance = extract(model_log_variance, t, x_t.shape)

        if self.gaussian_parametrization == 'eps':
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        elif self.gaussian_parametrization == 'x0':
            pred_xstart = model_output
        else:
            raise NotImplementedError
        
        # print('pred_xstart std', torch.std(pred_xstart))
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t,
        )
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape)

        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart,
        }
    
    def _vb_terms_bpd(self, model_output, x_start, x_t, t):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t,
        )
        out = self.gaussian_p_mean_variance(model_output, x_t, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out['mean'], out['log_variance'],
        )
        kl = mean_flat(kl) / np.log(2)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out['mean'], log_scales=0.5 * out['log_variance'],
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2)

        # at the first timestep return the decoder NLL
        # otherwise return $KL(q(x_{t - 1} | x_t, x_0) || p(x_{t - 1} | x_t))$
        output = torch.where((t == 0), decoder_nll, kl)
        out_mean = out['mean'] 
        return {
            'output': output, 
            'pred_xstart': out['pred_xstart'], 
            'out_mean': out_mean, 
            'true_mean': true_mean,
        }

    def _prior_gaussian(self, x_start: torch.Tensor):
        """Get the prior KL term for the variational lower-bound, measured in bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.
        
        Args:
            x_start: tensor of shape `[N, C, ...]`
        
        Returns:
            KL values in bits of shape `[N,]`
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.,
        )
        return mean_flat(kl_prior) / np.log(2)
    
    def _gaussian_loss(self, model_out, x_start, x_t, t, noise):
        if self.gaussian_loss_type == 'mse':
            loss = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == 'kl':
            loss = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
            )['output']
        return loss

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        answer = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        # print('coef', extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)[0, 0])
        # print('`x_t` std', torch.std(x_t), 'answer std', torch.std(answer))
        return answer
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        answer = (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return answer
    
    def gaussian_p_sample(
        self,
        model_out,
        x_t,
        t,
    ):
        # print('model_out_num std', torch.std(model_out))
        out = self.gaussian_p_mean_variance(
            model_out,
            x_t,
            t,
        )
        # no noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        sample = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        pred_xstart = out['pred_xstart']
        return {
            'sample': sample, 
            'pred_xstart': pred_xstart,
        }
    
    # multinomial part
    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    
    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        # $alpha_t * E[xt] + (1 - alpha_t) 1 / K$
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )
        return log_probs
    
    def predict_start(self, model_out, log_x_t):
        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f'{model_out.size()}'
        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred
    
    def q_posterior(self, log_x_start, log_x_t, t):
        t_minus_1 = t - 1
        # remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_ev_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)
        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_ev_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_ev_qxtmin_x0.to(torch.float32))
        unnormed_logprobs = log_ev_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        log_ev_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(unnormed_logprobs, self.offsets)
        return log_ev_xtmin_given_xt_given_xstart
    
    def p_pred(self, model_out, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(model_out, log_x)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t,
            )
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(model_out, log_x)
        else:
            raise ValueError
        return log_model_pred
    
    @torch.no_grad()
    def p_sample(self, model_out, log_x, t):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out
    
    def log_sample_categorical(self, logits):
        full_sample = []
        offset = 1e-30
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + offset) + offset)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_ev_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_ev_qxt_x0)
        return log_sample
    
    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxt_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes_expanded * torch.ones_like(log_qxt_prob))

        kl_prior = self.multinomial_kl(log_qxt_prob, log_half_prob)
        return sum_except_batch(kl_prior)
    
    def compute_lt(self, model_out, log_x_start, log_x_t, t, detach_mean=False):
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t,
        )
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt):
        if self.multinomial_loss_type == 'vb_stochastic':
            kl = self.compute_lt(model_out, log_x_start, log_x_t, t)
            kl_prior = self.kl_prior(log_x_start)
            # upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return vb_loss

        raise ValueError()

    def sample_time(self, b, device, method='uniform'):
        offset = 1e-10
        bias = 0.0001
        if method == 'importance':
            if not (self.lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')
            lt_sqrt = torch.sqrt(self.lt_history + offset) + bias
            lt_sqrt[0] = lt_sqrt[1]
            pt_all = (lt_sqrt / lt_sqrt.sum()).to(device)
            t = torch.multinomial(pt_all, n_samples=b, replacement=True).to(device)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        
        raise ValueError

    def mixed_loss(self, x, cond):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, 'uniform')

        # split numerical and categorical features
        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:]
        
        # process numerical and categorical features separately
        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)
        x_t = torch.cat([x_num_t, log_x_cat_t], dim=1)

        # denoise
        model_out = self._denoise_fn(x_t, t, cond)
        model_out_num = model_out[:, :self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features:]

        # compute losses
        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(model_out_cat, log_x_cat, log_x_cat_t, t, pt) / len(self.num_classes)
        if x_num.shape[1] > 0:
            loss_gauss = self._gaussian_loss(model_out_num, x_num, x_num_t, t, noise)
        return loss_multi.mean(), loss_gauss.mean()
    
    def sample(self, batch_size, c_dist: List[torch.Tensor], num_generated: int):
        with torch.no_grad():
            b = batch_size
            device = self.log_alpha.device
            z_norm = torch.randn((b, self.num_numerical_features), device=device)

            has_cat = self.num_classes[0] != 0
            log_z = torch.zeros((b, 0), device=device).float()
            if has_cat:
                uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
                log_z = self.log_sample_categorical(uniform_logits)

            columns = []
            for dist in c_dist:
                column = torch.multinomial(
                    dist,
                    num_samples=batch_size,
                    replacement=True,
                )
                columns.append(column)
            matrix = torch.stack(columns, dim=1)
            cond = matrix.long().to(device)
        
            if self.is_fair:
                instruction = cond
            else:
                instruction = cond[:, 0].unsqueeze(1)  # only use the first column which is the outcome
        
        x_t = torch.cat([z_norm, log_z], dim=1).float()
        for i in reversed(range(0, self.num_timesteps)):
            with torch.no_grad():
                if i != self.num_timesteps - 1:
                    print(f'sampling timestep {self.num_timesteps - 1:04d} -> {i:04d} -- generated: {num_generated:04d}', end='\r')
                else:
                    print(f'sampling timestep {self.num_timesteps - 1:04d} -> {i:04d} -- generated: {num_generated:04d}')
                t = torch.full((b,), i, device=device, dtype=torch.long)
                model_out = self._denoise_fn(
                    x_t,
                    t,
                    instruction,
                )
                model_out_num = model_out[:, :self.num_numerical_features]
                model_out_cat = model_out[:, self.num_numerical_features:]
                z_norm = self.gaussian_p_sample(model_out_num, z_norm, t)['sample']
                if has_cat:
                    log_z = self.p_sample(model_out_cat, log_z, t)
            x_t = torch.cat([z_norm, log_z], dim=1).float()
    
        with torch.no_grad():
            z_ohe = torch.exp(log_z).round()
            z_cat = log_z
            if has_cat:
                z_cat = ohe_to_categories(z_ohe, self.num_classes)
            sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, cond.cpu()

    def sample_all(self, n_samples: int, c_dist: List[torch.Tensor], batch_size: int = 1000):
        sample_fn = self.sample
        
        all_samples = []
        all_cond = []
        num_generated = 0
        while num_generated < n_samples:
            samples, cond = sample_fn(batch_size, c_dist, num_generated)
            mask_nam = torch.any(samples.isnan(), dim=1)
            samples = samples[~mask_nam]
            cond = cond[~mask_nam]
            all_samples.append(samples)
            all_cond.append(cond)
            if samples.shape[0] != batch_size:
                raise ValueError('found nan during sampling')
            num_generated += samples.shape[0]
        
        x_gen = torch.cat(all_samples, dim=0)[:n_samples]
        cond_gen = torch.cat(all_cond, dim=0)[:n_samples]
        
        return x_gen, cond_gen

    def update_guid_cfg(self, guid_cfg):
        self._denoise_fn.update_guid_cfg(guid_cfg)

def _test():
    num_diffusion_timesteps = 1000
    beta_schedule = get_named_beta_schedule('cosine', num_diffusion_timesteps)
    print(f'beta_schedule.shape: {beta_schedule.shape}')
    diffusion = GaussianMultinomialDiffusion(
        num_classes=np.array([2, 3, 4]),
        num_numerical_features=2,
        denoise_fn=lambda x, t, cond: x,
    )
    
    # mixed loss
    x_num = torch.randn(10, 2)  # numerical features
    x_cat = torch.randint(0, 1, (10, 9))  # categorical features: 9 features with 2, 3, 4 categories
    x = torch.cat([x_num, x_cat], dim=1)
    print(f'x.shape: {x.shape} after one-hot encoding')
    loss = diffusion.mixed_loss(
        x=x,
        cond=torch.randint(0, 2, (10,)),
    )
    print(f'mloss: {loss[0].item():.4f}, gloss: {loss[1].item():.4f}')
    
    # sample
    n_samples = 10
    c_dist = [torch.ones(2)]
    sample, cond = diffusion.sample(n_samples, c_dist, n_samples)
    print(f'sample.shape: {sample.shape} after ordinal encoding')
    print(f'cond.shape: {cond.shape}')

if __name__ == '__main__':
    _test()
