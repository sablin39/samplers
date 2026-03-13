from __future__ import annotations

import torch

from samplers.base import BaseSampler, PromptLike, SamplerOutput
from samplers.cache import BaseCache


def systematic_resample(
    normalized_weights: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    num_particles = normalized_weights.shape[0]
    positions = (
        torch.rand(1, device=normalized_weights.device, generator=generator) + torch.arange(
            num_particles,
            device=normalized_weights.device,
            dtype=normalized_weights.dtype,
        )
    ) / num_particles
    cumulative = torch.cumsum(normalized_weights, dim=0)
    cumulative[-1] = 1.0
    return torch.searchsorted(cumulative, positions, right=False).long()


class PowerSMCSampler(BaseSampler):
    def __init__(
        self,
        *args,
        alpha: float = 4.0,
        num_particles: int = 8,
        ess_threshold: float = 0.5,
        ramp_steps: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if alpha < 1.0:
            raise ValueError("alpha must be at least 1.")
        if num_particles < 2:
            raise ValueError("num_particles must be at least 2.")
        if not 0.0 < ess_threshold <= 1.0:
            raise ValueError("ess_threshold must lie in (0, 1].")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative.")
        self.alpha = alpha
        self.num_particles = num_particles
        self.ess_threshold = ess_threshold
        self.ramp_steps = ramp_steps

    @property
    def name(self) -> str:
        return "powersmc"

    def build_alpha_schedule(self, max_new_tokens: int) -> list[float]:
        if self.ramp_steps <= 0:
            return [self.alpha] * (max_new_tokens + 1)

        alpha_schedule = [1.0]
        for step in range(1, max_new_tokens + 1):
            ramp_fraction = min(step, self.ramp_steps) / self.ramp_steps
            alpha_schedule.append(1.0 + (self.alpha - 1.0) * ramp_fraction)
        return alpha_schedule

    def generate(
        self,
        prompt: PromptLike,
        *,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> SamplerOutput:
        rendered_prompt, encoded = self.encode_prompt(prompt)
        prompt_input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        repeated_prompt_ids = prompt_input_ids.repeat(self.num_particles, 1)
        repeated_attention_mask = None
        if attention_mask is not None:
            repeated_attention_mask = attention_mask.repeat(self.num_particles, 1)

        generator = self.make_generator(seed)
        alpha_schedule = self.build_alpha_schedule(max_new_tokens)

        cache = BaseCache(self.model)
        current_input_ids = repeated_prompt_ids
        generated_ids = torch.empty((self.num_particles, 0), dtype=torch.long, device=self.device)
        done = torch.zeros(self.num_particles, dtype=torch.bool, device=self.device)
        log_weights = torch.zeros(self.num_particles, dtype=torch.float64, device=self.device)
        log_prefix_probabilities = torch.zeros(self.num_particles, dtype=torch.float64, device=self.device)
        ess_history: list[float] = []
        resample_steps: list[int] = []

        with torch.inference_mode():
            for step in range(max_new_tokens):
                outputs = self._forward_next(
                    current_input_ids,
                    attention_mask=repeated_attention_mask,
                    cache=cache,
                )
                repeated_attention_mask = None

                logits = outputs.logits[:, -1, :].float()
                proposal_alpha = alpha_schedule[step]
                target_alpha = alpha_schedule[step + 1]
                base_log_probs = torch.log_softmax(logits, dim=-1)
                proposal_log_probs = torch.log_softmax(logits * proposal_alpha, dim=-1)

                sampled_tokens = torch.full(
                    (self.num_particles,),
                    self.eos_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                active = ~done
                if active.any():
                    proposal_probs = proposal_log_probs[active].exp()
                    sampled_tokens[active] = torch.multinomial(
                        proposal_probs,
                        num_samples=1,
                        generator=generator,
                    ).squeeze(-1)

                generated_ids = torch.cat([generated_ids, sampled_tokens.unsqueeze(-1)], dim=1)
                chosen_tokens = sampled_tokens.unsqueeze(-1)
                chosen_base_log_probs = base_log_probs.gather(1, chosen_tokens).squeeze(-1).to(torch.float64)
                chosen_proposal_log_probs = proposal_log_probs.gather(1, chosen_tokens).squeeze(-1).to(torch.float64)

                if active.any():
                    log_prefix_probabilities[active] += chosen_base_log_probs[active]
                    log_weights[active] += (
                        proposal_alpha * chosen_base_log_probs[active] - chosen_proposal_log_probs[active]
                    )
                    if target_alpha != proposal_alpha:
                        log_weights[active] += (target_alpha - proposal_alpha) * log_prefix_probabilities[active]

                done = done | sampled_tokens.eq(self.eos_token_id)
                current_input_ids = sampled_tokens.unsqueeze(-1)

                normalized_weights = self._normalized_weights(log_weights)
                ess = float((1.0 / torch.sum(normalized_weights.square())).item())
                ess_history.append(ess)

                if done.all():
                    break

                if ess < self.ess_threshold * self.num_particles and step + 1 < max_new_tokens:
                    ancestors = systematic_resample(normalized_weights, generator=generator)
                    generated_ids = generated_ids.index_select(0, ancestors)
                    done = done.index_select(0, ancestors)
                    log_prefix_probabilities = log_prefix_probabilities.index_select(0, ancestors)
                    current_input_ids = current_input_ids.index_select(0, ancestors)
                    cache.reorder(ancestors)
                    log_weights.zero_()
                    resample_steps.append(step + 1)

        final_weights = self._normalized_weights(log_weights)
        selected_index = int(
            torch.multinomial(final_weights.float(), num_samples=1, generator=generator).item()
        )
        selected_token_ids = self.trim_after_eos(generated_ids[selected_index].tolist())
        selected_log_probability = float(log_prefix_probabilities[selected_index].item())

        return SamplerOutput(
            sampler_name=self.name,
            prompt=self._prompt_text(prompt),
            rendered_prompt=rendered_prompt,
            prompt_token_ids=prompt_input_ids[0].tolist(),
            generated_token_ids=selected_token_ids,
            text=self.decode_token_ids(selected_token_ids),
            log_probability=selected_log_probability,
            metadata={
                "alpha": self.alpha,
                "num_particles": self.num_particles,
                "ess_threshold": self.ess_threshold,
                "ramp_steps": self.ramp_steps,
                "alpha_schedule": alpha_schedule,
                "ess_history": ess_history,
                "resample_steps": resample_steps,
                "final_particle_weights": final_weights.tolist(),
                "selected_particle": selected_index,
            },
        )

    def _normalized_weights(self, log_weights: torch.Tensor) -> torch.Tensor:
        shifted = log_weights - log_weights.max()
        return torch.softmax(shifted, dim=0)
