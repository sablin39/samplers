from __future__ import annotations

import torch

from samplers.base import BaseSampler, SamplerOutput


class GreedySampler(BaseSampler):
    @property
    def name(self) -> str:
        return "greedy"

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> SamplerOutput:
        return self._generate_single_path(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            seed=seed,
            token_selector=self._select_next_token,
            metadata={"strategy": "argmax"},
        )

    def _select_next_token(
        self,
        logits: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.LongTensor:
        del generator
        return logits.argmax(dim=-1, keepdim=True)


class StochasticSampler(BaseSampler):
    def __init__(
        self,
        *args,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "stochastic"

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> SamplerOutput:
        return self._generate_single_path(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            seed=seed,
            token_selector=self._select_next_token,
            metadata={"strategy": "categorical", "temperature": self.temperature},
        )

    def _select_next_token(
        self,
        logits: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.LongTensor:
        proposal_log_probs = torch.log_softmax(logits / self.temperature, dim=-1)
        proposal_probs = proposal_log_probs.exp()
        return torch.multinomial(proposal_probs, num_samples=1, generator=generator)
