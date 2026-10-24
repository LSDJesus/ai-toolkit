"""MOSS-Image sampling (training previews / inference engine).

Matches the vendor pipeline: shift-5 sigma grid snapped to the nearest of the
1000 training sigmas, fp32 Euler on fp32 noise, CFG 3 whose negative branch
skips DiT block 22 (two forwards per step), empty negative prompt. Flow
convention is the toolkit's: sigma 1 = noise, the DiT predicts noise - clean.
On request (the inference engine's sample.extra_kwargs) it samples instead
with ComfyUI's er_sde sampler and/or beta scheduler, ported step for step.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

DEFAULT_SHIFT = 5.0
DEFAULT_GUIDANCE = 3.0
DEFAULT_STEPS = 30
DEFAULT_UNCOND_SKIP_LAYERS = (22,)
NUM_TRAIN_TIMESTEPS = 1000


def moss_sigmas(
    num_steps: int,
    shift: float = DEFAULT_SHIFT,
    training_shift: float = 1.0,
    total: int = NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """The vendor scheduler's sigmas (MOSSImageScheduler.set_timesteps) with
    the terminal 0 appended: a shifted grid snapped to the training grid,
    consecutive duplicates removed (so fewer than num_steps are possible)."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


# ComfyUI's samplers, ported for the inference engine (sample.extra_kwargs
# {"sampler": "er_sde", "scheduler": "beta"}). They follow ComfyUI's own MOSS
# model sampling (ModelSamplingDiscreteFlow, shift 5, multiplier 1), not the
# vendor grid above, so a render matches the same choice in a KSampler.


def comfy_flow_sigmas(
    shift: float = DEFAULT_SHIFT, total: int = NUM_TRAIN_TIMESTEPS
) -> torch.Tensor:
    """ComfyUI's MOSS sigma table: the shifted sigma at k / total for
    k = 1..total, ascending."""
    t = torch.arange(1, total + 1, 1) / total
    return shift * t / (1 + (shift - 1) * t)


def beta_sigmas(
    num_steps: int,
    shift: float = DEFAULT_SHIFT,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> torch.Tensor:
    """ComfyUI's "beta" scheduler (comfy/samplers.py beta_scheduler) on its
    MOSS sigma table, the terminal 0 appended: timesteps at the Beta(alpha,
    beta) quantiles, repeats dropped."""
    import numpy
    import scipy.stats

    table = comfy_flow_sigmas(shift)
    total_timesteps = len(table) - 1
    ts = 1 - numpy.linspace(0, 1, num_steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    sigs, last_t = [], -1
    for t in ts:
        if t != last_t:
            sigs.append(float(table[int(t)]))
        last_t = t
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32)


def sample_er_sde(
    denoise,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    shift: float = DEFAULT_SHIFT,
    noise_seed: Optional[int] = None,
    s_noise: float = 1.0,
    max_stage: int = 3,
) -> torch.Tensor:
    """ComfyUI's er_sde (comfy/k_diffusion/sampling.py sample_er_sde, VP
    ER-SDE-Solver-3, arXiv 2309.06169) for a flow model, step for step.
    ``denoise(x, sigma, step)`` returns the x0 estimate. As in ComfyUI, a
    first sigma of 1 moves to the shifted sigma at 1 - 1e-4, lambda is
    sigma / (1 - sigma), the noise scaler is x * (exp(x^0.3) + 10), the
    integrals use 200 points, and each step's fresh noise comes from a
    generator on x's device seeded with ``noise_seed`` (+1 on CPU)."""
    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(noise_seed) + (1 if x.device.type == "cpu" else 0))

    def noise_scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    num_integration_points = 200.0
    point_indice = torch.arange(0, num_integration_points, dtype=torch.float32, device=x.device)
    sigmas = sigmas.clone()
    if len(sigmas) > 1 and sigmas[0] >= 1:
        t = 1.0 - 1e-4  # ComfyUI's percent_to_sigma(1e-4)
        sigmas[0] = shift * t / (1 + (shift - 1) * t)
    half_log_snrs = sigmas.logit().neg()
    er_lambdas = half_log_snrs.neg().exp()  # sigma / alpha

    old_denoised = old_denoised_d = None
    for i in range(len(sigmas) - 1):
        denoised = denoise(x, sigmas[i], i)
        stage_used = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            er_lambda_s, er_lambda_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / er_lambda_s
            alpha_t = sigmas[i + 1] / er_lambda_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)

            # stage 1: Euler
            x = r_alpha * r * x + alpha_t * (1 - r) * denoised

            if stage_used >= 2:
                dt = er_lambda_t - er_lambda_s
                lambda_step_size = -dt / num_integration_points
                lambda_pos = er_lambda_t + point_indice * lambda_step_size
                scaled_pos = noise_scaler(lambda_pos)

                # stage 2
                s = torch.sum(1 / scaled_pos) * lambda_step_size
                denoised_d = (denoised - old_denoised) / (er_lambda_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(er_lambda_t)) * denoised_d

                if stage_used >= 3:
                    # stage 3
                    s_u = torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                    denoised_u = (denoised_d - old_denoised_d) / ((er_lambda_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(er_lambda_t)) * denoised_u
                old_denoised_d = denoised_d

            if s_noise > 0:
                noise = torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)
                x = x + alpha_t * noise * s_noise * (er_lambda_t ** 2 - er_lambda_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x


def stack_text_embeds(
    embeds: List[torch.Tensor], device: torch.device, dtype: torch.dtype
):
    """Per-item (L_i, D) prompt states -> right-padded (B, S, D) + (B, S) mask."""
    length = max(e.shape[0] for e in embeds)
    stacked = torch.stack(
        [F.pad(e.to(device, dtype), (0, 0, 0, length - e.shape[0])) for e in embeds]
    )
    mask = torch.stack(
        [
            torch.arange(length, device=device) < e.shape[0]
            for e in embeds
        ]
    )
    return stacked, mask


class MossImagePipeline:
    """Lightweight sampler used by ai-toolkit's preview generation."""

    def __init__(self, model):
        # the MossImageModel, for its transformer + encode/decode
        self.model = model

    @property
    def device(self):
        return self.model.device_torch

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, **kwargs):
        pass

    @torch.no_grad()
    def __call__(
        self,
        conditional_embeds,
        unconditional_embeds,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        shift: float = DEFAULT_SHIFT,
        uncond_skip_layers: Sequence[int] = DEFAULT_UNCOND_SKIP_LAYERS,
        ref_latents: Optional[List[List[torch.Tensor]]] = None,
        isolate_refs: bool = False,
        sampler: str = "euler",
        scheduler: str = "moss",
        noise_seed: Optional[int] = None,
        sampler_options: Optional[dict] = None,
        **kwargs,
    ):
        """sampler "euler" (the vendor's) or "er_sde" (ComfyUI's, seeded by
        noise_seed); scheduler "moss" (the vendor grid) or "beta" (ComfyUI's).
        sampler_options: s_noise / max_stage for er_sde, beta_alpha /
        beta_beta for beta."""
        if sampler not in ("euler", "er_sde"):
            raise ValueError(f"unknown MOSS sampler {sampler!r} (euler | er_sde)")
        if scheduler not in ("moss", "beta"):
            raise ValueError(f"unknown MOSS scheduler {scheduler!r} (moss | beta)")
        opts = sampler_options or {}
        model = self.model
        device = model.device_torch
        dtype = model.torch_dtype
        transformer = model.transformer

        do_cfg = unconditional_embeds is not None and guidance_scale != 1.0
        cond, cond_mask = stack_text_embeds(conditional_embeds.text_embeds, device, dtype)
        if do_cfg:
            uncond, uncond_mask = stack_text_embeds(
                unconditional_embeds.text_embeds, device, dtype
            )

        shape = (1, transformer.params.latent_channels, height // 16, width // 16)
        if latents is None:
            # fp32 noise: bf16 noise promoted afterwards cannot regain precision
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        latents = latents.to(device, torch.float32)

        # edit: references ride in both branches. Isolated references get
        # their K/V computed on the first step and reused after; one cache per
        # branch (the reference RoPE follows the prompt length, and the
        # negative branch skips blocks)
        has_refs = ref_latents is not None and any(len(r) > 0 for r in ref_latents)
        ref_kwargs = {}
        cond_cache = uncond_cache = None
        if has_refs:
            ref_kwargs = {"ref_latents": ref_latents, "isolate_refs": isolate_refs}
            if isolate_refs:
                cond_cache, uncond_cache = {}, {}

        if scheduler == "beta":
            sigmas = beta_sigmas(
                num_inference_steps,
                shift=shift,
                alpha=float(opts.get("beta_alpha", 0.6)),
                beta=float(opts.get("beta_beta", 0.6)),
            ).to(device)
        else:
            sigmas = moss_sigmas(num_inference_steps, shift=shift).to(device)
        num_steps = len(sigmas) - 1

        def velocity(x, sigma):
            timestep = (sigma * NUM_TRAIN_TIMESTEPS).reshape(1)
            # CFG combines fp32 predictions even though the DiT runs in bf16
            pred = transformer(
                x, timestep, cond, cond_mask, ref_kv_cache=cond_cache, **ref_kwargs
            ).float()
            if do_cfg:
                pred_uncond = transformer(
                    x,
                    timestep,
                    uncond,
                    uncond_mask,
                    skip_layers=uncond_skip_layers,
                    ref_kv_cache=uncond_cache,
                    **ref_kwargs,
                ).float()
                pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            return pred

        def emit(x0, step_index):
            if getattr(model, "sample_step_hook", None) is not None:
                # inference engine preview: x0 estimate
                model._emit_sample_step(x0, step_index, num_steps)

        if sampler == "er_sde":

            def denoise(x, sigma, step_index):
                x0 = x - sigma * velocity(x, sigma)
                emit(x0, step_index)
                return x0

            latents = sample_er_sde(
                denoise,
                latents,
                sigmas,
                shift=shift,
                noise_seed=noise_seed,
                s_noise=float(opts.get("s_noise", 1.0)),
                max_stage=int(opts.get("max_stage", 3)),
            )
        else:
            for step_index in range(num_steps):
                sigma, sigma_next = sigmas[step_index], sigmas[step_index + 1]
                pred = velocity(latents, sigma)
                emit(latents - sigma * pred, step_index)
                latents = latents + (sigma_next - sigma) * pred

        images = model.decode_latents(latents, device=device, dtype=torch.float32)
        images = images.float().clamp(-1.0, 1.0)
        images = ((images + 1.0) * 127.5).round().to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(arr) for arr in images]
