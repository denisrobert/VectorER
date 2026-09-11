"""Construction + serialization for the Fellegi-Sunter scorer.

Holds the instance constructor (which also plants the reflexivity /
idempotence attribute) and the ``from_settings`` / ``from_comparisons`` /
``save`` / ``load``/ ``to_dict`` path.  ``_as_specs`` and ``_as_comparisons``
normalize arbitrary comparison inputs (``Comparison`` / ``ComparisonSpec`` /
dicts) into the internal shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..comparisons import Comparison, ComparisonSpec, comparison_from_dict
from ._constants import DEFAULT_PRIOR, DEFAULT_THRESHOLD, _LOG_CLIP
from ._weights import WeightTable


def _as_specs(comparisons: Sequence[Any]) -> list[ComparisonSpec]:
    out = []
    for item in comparisons:
        if isinstance(item, Comparison):
            out.append(item.spec())
        elif isinstance(item, ComparisonSpec):
            out.append(item)
        elif isinstance(item, dict):
            c = comparison_from_dict(item)
            out.append(c.spec() if c is not None else None)
        else:
            raise TypeError(f"expected Comparison / ComparisonSpec, got {type(item).__name__}")
    return [spec for spec in out if spec is not None]


def _as_comparisons(comparisons: Sequence[Any]) -> list[Comparison]:
    out = []
    for item in comparisons:
        if isinstance(item, Comparison):
            out.append(item)
        elif isinstance(item, ComparisonSpec):
            out.append(Comparison.from_spec(item))
        elif isinstance(item, dict):
            c = comparison_from_dict(item)
            if c is not None:
                out.append(c)
    return out


class ConstructionMixin:
    """Constructor, classmethod builders and serialization.

    Abstract: the concrete class combines this with the inference/training/
    candidate mixins; ``WeightTable`` is supplied by :func:`from_comparisons`
    / :func:`from_settings` unless the caller builds it directly.
    """

    def __init__(
        self,
        table: WeightTable,
        threshold: float = DEFAULT_THRESHOLD,
        comparisons: Optional[Sequence[Any]] = None,
        prior: Optional[float] = None,
        trained_settings: Optional[dict] = None,
        idempotent: bool = True,
    ) -> None:
        self.table = table
        self.threshold = float(threshold)
        self.comparisons = list(comparisons) if comparisons is not None else []
        self.prior = float(prior) if prior is not None else table.prior
        self.trained_settings = trained_settings
        self._em = (trained_settings or {}).get("_em")
        # Reflexivity (idempotence of the match function): content-identical
        # pairs are forced to posterior 1.0.  Without this, a "thin" record
        # (few non-null comparison fields) would score against itself below
        # the threshold (its null levels carry no evidence, leaving the prior),
        # breaking the r ~ r property required by the Swoosh Union-Class ICAR
        # construction.  Set False to restore the raw calibrated posterior.
        self.idempotent = bool(idempotent)
        self._prior_bf = (
            table.prior / (1.0 - table.prior) if table.prior != 1.0 else float("inf")
        )
        self._log_prior_bf = float(
            np.clip(np.log(self._prior_bf), -_LOG_CLIP, _LOG_CLIP)
        )

    @classmethod
    def from_comparisons(
        cls,
        comparisons: Sequence[Any],
        prior: float = DEFAULT_PRIOR,
        threshold: float = DEFAULT_THRESHOLD,
        base_records: Optional[Sequence[dict]] = None,
        idempotent: bool = True,
    ) -> "FellegiSunterScorer":
        """Build from declared comparisons (``Comparison`` or ``ComparisonSpec``)."""
        specs = _as_specs(comparisons)
        return cls(
            WeightTable(specs, prior=prior, base_records=base_records),
            threshold=threshold,
            comparisons=_as_comparisons(comparisons),
            prior=prior,
            idempotent=idempotent,
        )

    @classmethod
    def from_settings(
        cls,
        settings: dict,
        threshold: float = DEFAULT_THRESHOLD,
        base_records: Optional[Sequence[dict]] = None,
        idempotent: bool = True,
    ) -> "FellegiSunterScorer":
        """Build from a resolved settings dict (trained m/u).

        ``settings["comparisons"]`` is a list of ``{"type", "params", "levels"}``
        entries as produced by :meth:`to_settings`.
        """
        comparisons = []
        for entry in settings.get("comparisons") or []:
            if isinstance(entry, dict) and "type" in entry:
                if "levels" in entry:
                    comparisons.append(Comparison.from_resolved(entry))
                else:
                    comparisons.append(Comparison.from_dict(entry))
            else:
                comparisons.append(entry)
        prior = settings.get("probability_two_random_records_match", DEFAULT_PRIOR)
        return cls(
            WeightTable(_as_specs(comparisons), prior=prior, base_records=base_records),
            threshold=threshold,
            comparisons=comparisons,
            prior=prior,
            trained_settings=settings,
            idempotent=settings.get("idempotent", idempotent),
        )

    def to_settings(self) -> dict:
        """Serializable settings (comparisons + current m/u + prior)."""
        return {
            "comparisons": [c.resolved() for c in self.comparisons],
            "probability_two_random_records_match": self.prior,
            "idempotent": self.idempotent,
            **({"_em": self._em} if self._em else {}),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            **self.to_settings(),
        }

    def save(self, path: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Any) -> "FellegiSunterScorer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_settings(
            data,
            threshold=float(data.get("threshold", DEFAULT_THRESHOLD)),
        )