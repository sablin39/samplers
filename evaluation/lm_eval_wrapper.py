"""``lm_eval`` wrapper for benchmarking samplers on standard LLM evaluation tasks.

Wraps any :class:`~samplers.base.BaseSampler` as an :class:`lm_eval.api.model.LM`
so that it can be evaluated on both **completion** and **chat-completion** benchmarks
via the `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_.

Usage examples::

    # Greedy sampler on HellaSwag (completion task)
    python -m evaluation.lm_eval_wrapper \
        --model-path Qwen/Qwen3-0.5B \
        --sampler greedy \
        --tasks hellaswag \
        --limit 100

    # Stochastic sampler with chat template (chat-completion task)
    python -m evaluation.lm_eval_wrapper \
        --model-path Qwen/Qwen3-0.5B \
        --sampler stochastic --temperature 0.8 \
        --tasks mmlu \
        --apply-chat-template

    # Save generated samples for later analysis
    python -m evaluation.lm_eval_wrapper \
        --model-path Qwen/Qwen3-0.5B \
        --sampler greedy \
        --tasks gsm8k \
        --output results/gsm8k_greedy
"""

from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval.api.model
import lm_eval.evaluator
from lm_eval import utils as lm_eval_utils
from lm_eval.api.instance import Instance

from samplers.base import BaseSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SamplerLM – lm_eval LM wrapper around any BaseSampler
# ---------------------------------------------------------------------------


class SamplerLM(lm_eval.api.model.LM):
    """Wraps a :class:`BaseSampler` to expose the ``lm_eval`` ``LM`` interface.

    * ``loglikelihood`` and ``loglikelihood_rolling`` run direct model forward
      passes (independent of the sampler) so that perplexity-style benchmarks
      work correctly.
    * ``generate_until`` delegates to the sampler's ``generate`` method and
      post-processes stop sequences.
    """

    def __init__(
        self,
        sampler: BaseSampler,
        *,
        max_length: int | None = None,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._sampler = sampler
        self._model = sampler.model
        self._tokenizer = sampler.tokenizer
        self._device = sampler.device
        self._max_length = max_length or getattr(
            self._model.config, "max_position_embeddings", 2048
        )
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size
        self._seed = seed
        self._seed_counter = 0

    # -- Properties -----------------------------------------------------------

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer.name_or_path

    @property
    def eot_token_id(self) -> int:
        return self._sampler.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    # -- Chat template support ------------------------------------------------

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self._tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        if not chat_template:
            return None
        template = getattr(self._tokenizer, "chat_template", None)
        if template is not None:
            return template if isinstance(template, str) else ""
        return ""

    # -- Seed management ------------------------------------------------------

    def _next_seed(self) -> int | None:
        if self._seed is None:
            return None
        seed = self._seed + self._seed_counter
        self._seed_counter += 1
        return seed

    # -- Tokenization helpers -------------------------------------------------

    def _tok_encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def _tok_decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    # -- loglikelihood --------------------------------------------------------

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="loglikelihood"):
            context, continuation = request.args
            result = self._loglikelihood_single(context, continuation)
            results.append(result)
            self.cache_hook.add_partial("loglikelihood", request.args, result)
        return results

    def _loglikelihood_single(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        if context == "":
            context_enc = [self.eot_token_id]
            continuation_enc = self._tok_encode(continuation, add_special_tokens=False)
        else:
            # Encode together then split for proper tokenization at boundary
            n_spaces = len(context) - len(context.rstrip())
            if n_spaces > 0:
                continuation = context[-n_spaces:] + continuation
                context = context[:-n_spaces]

            whole_enc = self._tok_encode(context + continuation)
            context_enc = self._tok_encode(context)
            continuation_enc = whole_enc[len(context_enc) :]

        if not continuation_enc:
            return (0.0, True)

        # Truncate from the left if exceeding max_length
        input_ids = context_enc + continuation_enc
        num_cont = len(continuation_enc)
        if len(input_ids) > self._max_length:
            input_ids = input_ids[-self._max_length :]
            # Ensure at least 1 context token remains
            num_cont = min(num_cont, len(input_ids) - 1)
            if num_cont <= 0:
                return (0.0, True)

        input_tensor = torch.tensor([input_ids], device=self._device)

        with torch.inference_mode():
            logits = self._model(input_ids=input_tensor).logits[0].float()

        # logits[i] predicts token at position i+1.
        # Continuation tokens occupy positions [-num_cont:].
        # Logits predicting them are at positions [-(num_cont+1):-1].
        cont_start = len(input_ids) - num_cont
        pred_logits = logits[cont_start - 1 : len(input_ids) - 1]
        log_probs = torch.log_softmax(pred_logits, dim=-1)

        cont_tensor = torch.tensor(
            input_ids[-num_cont:], dtype=torch.long, device=self._device
        )
        token_log_probs = log_probs.gather(1, cont_tensor.unsqueeze(1)).squeeze(1)
        total_log_prob = float(token_log_probs.sum().item())

        greedy_tokens = pred_logits.argmax(dim=-1)
        is_greedy = bool((greedy_tokens == cont_tensor).all().item())

        return (total_log_prob, is_greedy)

    # -- loglikelihood_rolling ------------------------------------------------

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        results: list[float] = []
        for request in tqdm(
            requests, disable=disable_tqdm, desc="loglikelihood_rolling"
        ):
            (string,) = request.args
            result = self._loglikelihood_rolling_single(string)
            results.append(result)
            self.cache_hook.add_partial(
                "loglikelihood_rolling", request.args, result
            )
        return results

    def _loglikelihood_rolling_single(self, string: str) -> float:
        token_ids = self._tok_encode(string)
        if not token_ids:
            return 0.0

        # Use lm_eval's rolling-window utilities for correct chunking.
        windows = list(
            map(
                lm_eval_utils.make_disjoint_window,
                lm_eval_utils.get_rolling_token_windows(
                    token_list=token_ids,
                    prefix_token=self.eot_token_id,
                    max_seq_len=self._max_length,
                    context_len=1,
                ),
            )
        )

        total_log_prob = 0.0
        for context_enc, continuation_enc in windows:
            if not continuation_enc:
                continue
            input_ids = context_enc + continuation_enc
            input_tensor = torch.tensor([input_ids], device=self._device)
            num_cont = len(continuation_enc)

            with torch.inference_mode():
                logits = self._model(input_ids=input_tensor).logits[0].float()

            cont_start = len(input_ids) - num_cont
            pred_logits = logits[cont_start - 1 : len(input_ids) - 1]
            log_probs = torch.log_softmax(pred_logits, dim=-1)
            cont_tensor = torch.tensor(
                continuation_enc, dtype=torch.long, device=self._device
            )
            token_log_probs = log_probs.gather(1, cont_tensor.unsqueeze(1)).squeeze(1)
            total_log_prob += float(token_log_probs.sum().item())

        return total_log_prob

    # -- generate_until -------------------------------------------------------

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        results: list[str] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="generate_until"):
            context, gen_kwargs = request.args
            result = self._generate_until_single(context, gen_kwargs)
            results.append(result)
            self.cache_hook.add_partial("generate_until", request.args, result)
        return results

    def _generate_until_single(self, context: str, gen_kwargs: dict[str, Any]) -> str:
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        max_gen_toks = gen_kwargs.get("max_gen_toks", self._max_gen_toks)

        # Truncate context from the left if it would leave no room for generation
        context_ids = self._tok_encode(context)
        max_ctx = self._max_length - max_gen_toks
        if max_ctx > 0 and len(context_ids) > max_ctx:
            context_ids = context_ids[-max_ctx:]
            context = self._tok_decode(context_ids)

        output = self._sampler.generate(
            context, max_new_tokens=max_gen_toks, seed=self._next_seed()
        )
        # Decode directly from token IDs to preserve leading whitespace.
        # BaseSampler.decode_token_ids applies .strip() which destroys
        # indentation critical for code generation tasks like HumanEval.
        generated = self._tokenizer.decode(
            output.generated_token_ids, skip_special_tokens=True
        )

        # Truncate at first stop sequence
        for stop_seq in until:
            idx = generated.find(stop_seq)
            if idx != -1:
                generated = generated[:idx]

        return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_sampler(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
) -> BaseSampler:
    """Construct a sampler from CLI arguments."""
    from samplers.baselines import GreedySampler, StochasticSampler
    from samplers.power_smc import PowerSMCSampler

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    common: dict[str, Any] = dict(
        model=model, tokenizer=tokenizer, device=device, use_chat_template=False
    )

    name = args.sampler
    if name == "greedy":
        return GreedySampler(**common)
    elif name == "stochastic":
        return StochasticSampler(**common, temperature=args.temperature)
    elif name == "powersmc":
        return PowerSMCSampler(
            **common,
            alpha=args.alpha,
            num_particles=args.particles,
            ess_threshold=args.ess_threshold,
            ramp_steps=args.ramp_steps,
        )
    else:
        raise ValueError(f"Unknown sampler: {name!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a sampler on lm-evaluation-harness benchmarks.",
    )
    # Model
    parser.add_argument(
        "--model-path", required=True, help="HuggingFace model name or local path."
    )
    parser.add_argument("--device", default=None, help="PyTorch device (e.g. cuda:0).")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    # Sampler
    parser.add_argument(
        "--sampler",
        default="greedy",
        choices=["greedy", "stochastic", "powersmc"],
        help="Sampler to use.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Stochastic sampler temperature."
    )
    parser.add_argument(
        "--alpha", type=float, default=4.0, help="PowerSMC power exponent."
    )
    parser.add_argument(
        "--particles", type=int, default=8, help="PowerSMC number of particles."
    )
    parser.add_argument(
        "--ess-threshold", type=float, default=0.5, help="PowerSMC ESS resampling threshold."
    )
    parser.add_argument(
        "--ramp-steps", type=int, default=0, help="PowerSMC alpha ramp steps."
    )
    # Tasks
    parser.add_argument(
        "--tasks", required=True, help="Comma-separated list of lm_eval task names."
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=None, help="Number of few-shot examples."
    )
    # Generation
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=256,
        help="Maximum tokens to generate per request.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    # Evaluation
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    # Chat template
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply the tokenizer's chat template to prompts.",
    )
    # Output
    parser.add_argument(
        "--output", default=None, help="Directory to save results and samples."
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        default=True,
        help="Save per-sample predictions (default: True).",
    )
    parser.add_argument(
        "--no-log-samples",
        action="store_false",
        dest="log_samples",
        help="Do not save per-sample predictions.",
    )
    # Code execution
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Allow code execution for tasks like HumanEval.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )

    sampler = _build_sampler(args, model, tokenizer)

    lm = SamplerLM(
        sampler,
        max_gen_toks=args.max_gen_toks,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Set up evaluation tracker for saving results
    evaluation_tracker = None
    if args.output:
        from lm_eval.loggers import EvaluationTracker

        evaluation_tracker = EvaluationTracker(output_path=args.output)

    task_list = [t.strip() for t in args.tasks.split(",")]

    seed = args.seed if args.seed is not None else 0

    results = lm_eval.evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        apply_chat_template=args.apply_chat_template,
        confirm_run_unsafe_code=args.allow_code_execution,
        random_seed=seed,
        numpy_random_seed=seed + 1234,
        torch_random_seed=seed + 1234,
    )

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if evaluation_tracker is not None:
            evaluation_tracker.save_results_aggregated(
                results=results, samples=results.get("samples", {})
            )

        # Save a concise summary JSON
        summary_path = output_dir / "results.json"
        serializable = {k: v for k, v in results.items() if k != "samples"}
        summary_path.write_text(json.dumps(serializable, indent=2, default=str))
        logger.info("Results saved to %s", summary_path)

        # Save samples separately if requested
        if args.log_samples and "samples" in results:
            samples_path = output_dir / "samples.json"
            samples_path.write_text(
                json.dumps(results["samples"], indent=2, default=str)
            )
            logger.info("Samples saved to %s", samples_path)

    # Print summary table
    from lm_eval.utils import make_table

    print(make_table(results))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
