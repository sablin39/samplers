from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoTokenizer

from samplers import PowerSMCSampler, StochasticSampler


DEFAULT_MODEL_PATH = "/home/rwkv/models/Qwen3.5-0.8B"
DEFAULT_OUTPUT_PATH = "evaluation/results/qwen3_5_0_8b_diversity.json"
DEFAULT_PROMPTS = [
    "Write a four-line poem about debugging at midnight.",
    "Invent a believable but unusual startup idea in three sentences.",
    "Describe a city where it rains music instead of water in one paragraph.",
    "Give five creative uses for a paperclip as a short bullet list.",
    "Tell a two-sentence science fiction microstory about a time-traveling botanist.",
]


def load_model_and_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = None
    load_errors: list[str] = []

    for model_class in (AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM):
        try:
            model = model_class.from_pretrained(model_path, dtype=dtype)
            break
        except Exception as exc:
            load_errors.append(f"{model_class.__name__}: {exc}")

    if model is None:
        error_text = "\n".join(load_errors)
        raise RuntimeError(f"Unable to load model at {model_path}.\n{error_text}")

    model.eval().to(device)
    return tokenizer, model


def tokenize_texts(tokenizer, texts: list[str]) -> list[list[int]]:
    tokenized: list[list[int]] = []
    for text in texts:
        tokenized.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    return tokenized


def distinct_n(tokenized_texts: list[list[int]], n: int) -> float:
    unique_ngrams: set[tuple[int, ...]] = set()
    total_ngrams = 0
    for token_ids in tokenized_texts:
        if len(token_ids) < n:
            continue
        ngrams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
        unique_ngrams.update(ngrams)
        total_ngrams += len(ngrams)
    if total_ngrams == 0:
        return 0.0
    return len(unique_ngrams) / total_ngrams


def summarize_texts(tokenizer, texts: list[str]) -> dict[str, Any]:
    normalized_texts = [text.strip() for text in texts]
    tokenized_texts = tokenize_texts(tokenizer, normalized_texts)
    total_tokens = sum(len(token_ids) for token_ids in tokenized_texts)
    unique_texts = len(set(normalized_texts))
    return {
        "num_samples": len(normalized_texts),
        "unique_rate": unique_texts / len(normalized_texts) if normalized_texts else 0.0,
        "distinct_1": distinct_n(tokenized_texts, 1),
        "distinct_2": distinct_n(tokenized_texts, 2),
        "avg_tokens": total_tokens / len(tokenized_texts) if tokenized_texts else 0.0,
    }


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("temperature", "alpha", "num_particles", "ess_threshold", "ramp_steps", "resample_steps"):
        if key in metadata:
            compact[key] = metadata[key]
    ess_history = metadata.get("ess_history")
    if ess_history:
        compact["mean_ess"] = float(sum(ess_history) / len(ess_history))
        compact["min_ess"] = float(min(ess_history))
    return compact


def run_diversity_experiment(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer, model = load_model_and_tokenizer(args.model_path, device=args.device)

    samplers = {
        "stochastic": StochasticSampler(
            model,
            tokenizer,
            device=args.device,
            temperature=args.temperature,
        ),
        "powersmc": PowerSMCSampler(
            model,
            tokenizer,
            device=args.device,
            alpha=args.alpha,
            num_particles=args.particles,
            ess_threshold=args.ess_threshold,
            ramp_steps=args.ramp_steps,
        ),
    }

    results: dict[str, Any] = {
        "model_path": args.model_path,
        "device": args.device,
        "samples_per_prompt": args.samples_per_prompt,
        "max_new_tokens": args.max_new_tokens,
        "prompts": DEFAULT_PROMPTS,
        "samplers": {},
    }

    for sampler_name, sampler in samplers.items():
        start_time = time.perf_counter()
        all_texts: list[str] = []
        prompt_results: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(DEFAULT_PROMPTS):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(rendered, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

            outputs = []
            for sample_index in range(args.samples_per_prompt):
                seed = args.seed + prompt_index * 100 + sample_index
                result = sampler.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                )
                text = tokenizer.decode(result.generated_ids.tolist(), skip_special_tokens=True)
                outputs.append(
                    {
                        "seed": seed,
                        "text": text,
                        "num_generated_tokens": result.num_generated_tokens,
                        "log_probability": result.log_probability,
                        "metadata": compact_metadata(result.metadata),
                    }
                )
                all_texts.append(text)

            prompt_texts = [item["text"] for item in outputs]
            prompt_results.append(
                {
                    "prompt": prompt,
                    "metrics": summarize_texts(tokenizer, prompt_texts),
                    "outputs": outputs,
                }
            )

        elapsed_seconds = time.perf_counter() - start_time
        aggregate_metrics = summarize_texts(tokenizer, all_texts)
        results["samplers"][sampler_name] = {
            "elapsed_seconds": elapsed_seconds,
            "aggregate_metrics": aggregate_metrics,
            "per_prompt": prompt_results,
        }

        print(
            f"{sampler_name:10s} "
            f"unique={aggregate_metrics['unique_rate']:.3f} "
            f"distinct1={aggregate_metrics['distinct_1']:.3f} "
            f"distinct2={aggregate_metrics['distinct_2']:.3f} "
            f"avg_tokens={aggregate_metrics['avg_tokens']:.1f} "
            f"time={elapsed_seconds:.1f}s"
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate diversity for stochastic sampling and Power-SMC.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--ramp-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA because the repository instructions forbid CPU fallback.")

    results = run_diversity_experiment(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
