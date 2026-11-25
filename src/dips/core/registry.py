"""Central operation registry for DIPS."""

from __future__ import annotations

from .types import OperationCategory, OperationSpec
from . import (
    ops_ai,
    ops_basics,
    ops_affine,
    ops_geometry,
    ops_intensity,
    ops_spatial,
    ops_frequency,
    ops_histogram,
    ops_morphology,
    ops_restoration,
    ops_color,
    ops_artistic,
    ops_features,
)


class OperationRegistry:
    """Ordered catalog of all registered DIP operations."""

    def __init__(self) -> None:
        self._specs: dict[str, OperationSpec] = {}
        self._order: list[str] = []
        for module in (
            ops_ai,
            ops_basics,
            ops_affine,
            ops_geometry,
            ops_intensity,
            ops_spatial,
            ops_frequency,
            ops_histogram,
            ops_morphology,
            ops_restoration,
            ops_color,
            ops_artistic,
            ops_features,
        ):
            for spec in module.get_specs():
                self.register(spec)

    def register(self, spec: OperationSpec) -> None:
        if spec.op_id not in self._specs:
            self._order.append(spec.op_id)
        self._specs[spec.op_id] = spec

    def get(self, op_id: str) -> OperationSpec:
        return self._specs[op_id]

    def __contains__(self, op_id: str) -> bool:
        return op_id in self._specs

    def all_specs(self) -> list[OperationSpec]:
        return [self._specs[k] for k in self._order]

    def categories(self) -> list[OperationCategory]:
        return list(OperationCategory)

    def by_category(self, category: OperationCategory) -> list[OperationSpec]:
        return [self._specs[k] for k in self._order if self._specs[k].category == category]

    def search(self, query: str) -> list[OperationSpec]:
        q = query.strip().lower()
        if not q:
            return self.all_specs()
        results: list[OperationSpec] = []
        for spec in self.all_specs():
            haystack = " ".join(
                (spec.name, spec.category.value, spec.description, " ".join(spec.keywords))
            ).lower()
            if q in haystack:
                results.append(spec)
        return results


_DEFAULT_REGISTRY: OperationRegistry | None = None


def get_registry() -> OperationRegistry:
    """Return the singleton OperationRegistry instance."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = OperationRegistry()
    return _DEFAULT_REGISTRY
