import functools
import torch
import numpy as np
import abc
from scipy import integrate
from baselines.stasy.models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
import baselines.stasy.sde_lib as sde_lib
from baselines.stasy.models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}

def register_predictor(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls
    if cls is None:
        return _register
    else:
        return _register(cls)

def get_predictor(name):
    return _PREDICTORS[name]

def get_corrector(name):
    return _CORRECTORS[name]

def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    sampler_name = config.sampling.method
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device
        )
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
        )
    else:
        raise ValueError(f'sampler name {sampler_name} unknown.')
    return sampling_fn

class Predictor(abc.ABC):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

class Corrector(abc.ABC):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None] * np.sqrt(-dt) * z
        return x, x_mean

@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None] * z
        return x, x_mean

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')
        assert not probability_flow, 'probability flow not supported by ancestral sampling'

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep].to(t.device)
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)

        x_mean = (x + beta[:, None] * score) / torch.sqrt(1. - beta)[:, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None] * noise

        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(x, t)

@register_predictor(name='none')
class NonePredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE) and not isinstance(sde, SubVPSDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size[:, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise

        return x, x_mean

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE) and not isinstance(sde, SubVPSDE):
            raise NotImplementedError(f'SDE class {sde.__class__.__name__} not yet supported')

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, SubVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None]

        return x, x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)

def get_pc_sampler(
    sde, shape, predictor, corrector, inverse_scaler, snr,
    n_steps=1, probability_flow=False, continuous=False,
    denoise=True, eps=1e-3, device='cuda',
):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(model, sampling_shape=None):
        if sampling_shape:
            shape = sampling_shape

        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler

def get_ode_sampler(
    sde, shape, inverse_scaler,
    denoise=False, rtol=1e-5, atol=1e-5,
    method='RK45', eps=1e-3, device='cuda',
):

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None, sampling_shape=None):
        if sampling_shape:
            shape = sampling_shape
        
        with torch.no_grad():
            if z is None:
                x = sde.prior_sampling(shape).to(device)
                start = eps
                end = sde.T
            else:
                shape = z.shape
                x = z
                start = sde.T
                end = sde.T + 1e-08

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            solution = integrate.solve_ivp(ode_func, (end, start), to_flattened_numpy(x), rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
            if denoise:
                x = denoise_update_fn(model, x)
            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler