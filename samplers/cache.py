from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


class BaseCache:
    """Unified cache wrapper for KV-cache state across model architectures."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._past_key_values: Any = None

    @property
    def past_key_values(self) -> Any:
        return self._past_key_values

    def update(self, model_output: Any) -> None:
        """Capture cache from a model forward output."""
        self._past_key_values = model_output.past_key_values

    def reorder(self, ancestors: torch.LongTensor) -> None:
        """Reorder cache in-place according to ancestor indices."""
        if self._past_key_values is None:
            return

        pkv = self._past_key_values

        if hasattr(self._model, "_reorder_cache"):
            reordered = self._model._reorder_cache(pkv, ancestors)
            if reordered is not None:
                self._past_key_values = reordered
            return

        if hasattr(pkv, "reorder_cache"):
            pkv.reorder_cache(ancestors)
            return

        if hasattr(pkv, "batch_select_indices"):
            pkv.batch_select_indices(ancestors)
            return

        self._past_key_values = self._recursive_index_select(pkv, ancestors)

    def __bool__(self) -> bool:
        return self._past_key_values is not None

    @staticmethod
    def _recursive_index_select(value: Any, ancestors: torch.LongTensor) -> Any:
        if isinstance(value, torch.Tensor):
            if value.shape[0] != ancestors.shape[0]:
                return value
            return value.index_select(0, ancestors)
        if isinstance(value, tuple):
            return tuple(BaseCache._recursive_index_select(item, ancestors) for item in value)
        if isinstance(value, list):
            return [BaseCache._recursive_index_select(item, ancestors) for item in value]
        if isinstance(value, Mapping):
            return type(value)(
                (key, BaseCache._recursive_index_select(item, ancestors)) for key, item in value.items()
            )
        if hasattr(value, "__dict__"):
            cloned = value.__class__.__new__(value.__class__)
            cloned.__dict__.update(
                {key: BaseCache._recursive_index_select(item, ancestors) for key, item in value.__dict__.items()}
            )
            return cloned
        return value
