from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from samplers.cache import BaseCache


PromptLike = str | Sequence[dict[str, Any]]


@dataclass
class SamplerOutput:
    sampler_name: str
    prompt: str
    rendered_prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    text: str
    log_probability: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)


class BaseSampler(ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str | torch.device | None = None,
        use_chat_template: bool = True,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else self._infer_device()
        self.use_chat_template = use_chat_template
        self.eos_token_id = self._resolve_token_id(
            self._candidate_eos_token_ids(),
            "Could not resolve an EOS token id from the model or tokenizer.",
        )
        self.pad_token_id = self._resolve_token_id(
            self._candidate_pad_token_ids(),
            "Could not resolve a PAD token id from the model or tokenizer.",
            default=self.eos_token_id,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: PromptLike,
        *,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> SamplerOutput:
        raise NotImplementedError

    def batch_generate(
        self,
        prompts: Sequence[PromptLike],
        *,
        max_new_tokens: int = 64,
        seed: int | None = None,
    ) -> list[SamplerOutput]:
        outputs: list[SamplerOutput] = []
        for index, prompt in enumerate(prompts):
            prompt_seed = None if seed is None else seed + index
            outputs.append(self.generate(prompt, max_new_tokens=max_new_tokens, seed=prompt_seed))
        return outputs

    def encode_prompt(
        self,
        prompt: PromptLike,
        *,
        add_generation_prompt: bool = True,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        rendered_prompt = self.render_prompt(prompt, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(rendered_prompt, return_tensors="pt")
        return rendered_prompt, dict(encoded)

    def render_prompt(self, prompt: PromptLike, *, add_generation_prompt: bool = True) -> str:
        if isinstance(prompt, str):
            if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            return prompt

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                list(prompt),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        content = []
        for message in prompt:
            role = str(message.get("role", "user"))
            body = str(message.get("content", ""))
            content.append(f"{role}: {body}")
        return "\n".join(content)

    def decode_token_ids(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()

    def trim_after_eos(self, token_ids: Sequence[int]) -> list[int]:
        trimmed: list[int] = []
        for token_id in token_ids:
            if int(token_id) == self.eos_token_id:
                break
            trimmed.append(int(token_id))
        return trimmed

    def make_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        return generator

    def _forward_next(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        cache: BaseCache | None = None,
    ) -> Any:
        model_inputs: dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
        if attention_mask is not None and (cache is None or not cache):
            model_inputs["attention_mask"] = attention_mask
        if cache and cache.past_key_values is not None:
            model_inputs["past_key_values"] = cache.past_key_values
        outputs = self.model(**model_inputs)
        if cache is not None:
            cache.update(outputs)
        return outputs

    def _generate_single_path(
        self,
        prompt: PromptLike,
        *,
        max_new_tokens: int,
        seed: int | None,
        token_selector: Callable[[torch.Tensor, torch.Generator | None], torch.LongTensor],
        metadata: dict[str, Any] | None = None,
    ) -> SamplerOutput:
        rendered_prompt, encoded = self.encode_prompt(prompt)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generator = self.make_generator(seed)
        cache = BaseCache(self.model)
        current_input_ids = input_ids
        selected_token_ids: list[int] = []
        selected_log_probs: list[float] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self._forward_next(
                    current_input_ids,
                    attention_mask=attention_mask,
                    cache=cache,
                )
                logits = outputs.logits[:, -1, :].float()
                next_token = token_selector(logits, generator).view(1, 1)
                next_log_prob = torch.log_softmax(logits, dim=-1).gather(1, next_token)

                token_id = int(next_token.item())
                selected_token_ids.append(token_id)
                selected_log_probs.append(float(next_log_prob.item()))
                current_input_ids = next_token.to(self.device)

                if token_id == self.eos_token_id:
                    break

        trimmed_token_ids = self.trim_after_eos(selected_token_ids)
        base_log_probability = float(sum(selected_log_probs)) if selected_log_probs else 0.0
        return SamplerOutput(
            sampler_name=self.name,
            prompt=self._prompt_text(prompt),
            rendered_prompt=rendered_prompt,
            prompt_token_ids=input_ids[0].tolist(),
            generated_token_ids=trimmed_token_ids,
            text=self.decode_token_ids(trimmed_token_ids),
            log_probability=base_log_probability,
            metadata=metadata or {},
        )

    def _candidate_eos_token_ids(self) -> list[Any]:
        config = getattr(self.model, "config", None)
        generation_config = getattr(self.model, "generation_config", None)
        return [
            getattr(getattr(config, "text_config", None), "eos_token_id", None),
            getattr(config, "eos_token_id", None),
            getattr(generation_config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        ]

    def _candidate_pad_token_ids(self) -> list[Any]:
        config = getattr(self.model, "config", None)
        generation_config = getattr(self.model, "generation_config", None)
        return [
            getattr(getattr(config, "text_config", None), "pad_token_id", None),
            getattr(config, "pad_token_id", None),
            getattr(generation_config, "pad_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ]

    def _resolve_token_id(
        self,
        candidates: Sequence[Any],
        error_message: str,
        *,
        default: int | None = None,
    ) -> int:
        for candidate in candidates:
            token_id = self._coerce_token_id(candidate)
            if token_id is not None:
                return token_id
        if default is not None:
            return default
        raise ValueError(error_message)

    def _coerce_token_id(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, (list, tuple)) and value:
            first_value = value[0]
            return int(first_value) if first_value is not None else None
        return None

    def _infer_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            return torch.device("cpu")
        return parameter.device

    def _prompt_text(self, prompt: PromptLike) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n".join(f"{message.get('role', 'user')}: {message.get('content', '')}" for message in prompt)
