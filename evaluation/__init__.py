from __future__ import annotations

from typing import Any

__all__ = ["run_diversity_experiment"]


def run_diversity_experiment(*args: Any, **kwargs: Any):
    from evaluation.diversity import run_diversity_experiment as _run_diversity_experiment

    return _run_diversity_experiment(*args, **kwargs)
