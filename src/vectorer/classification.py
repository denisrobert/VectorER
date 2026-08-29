"""Classification stage: turn Fellegi-Sunter posteriors into decisions.

The classic Fellegi-Sunter decision rule partitions pairs into two or three
regions by thresholds on the match score.  :class:`ThresholdClassifier`
implements the two-band (match / non-match) rule used by the default pipelines
and an optional three-band rule (match / possible match / non-match) for
conservative pro-active linkage.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


class Decision(enum.Enum):
    """FS classification outcome for a scored candidate pair."""

    MATCH = "match"
    POSSIBLE_MATCH = "possible_match"
    NON_MATCH = "non_match"


@dataclass
class ScoredCandidate(Generic[T]):
    """A candidate record with its FS posterior and evidence."""

    record: T
    probability: float
    match_weight: float
    blocking_score: float
    position: int


@dataclass
class MatchResult(Generic[T]):
    """A decided match: the record, its posterior and evidence."""

    record: T
    match_probability: float
    match_weight: float
    blocking_score: float
    candidate_position: int


class Classifier:
    """Abstract stage mapping posteriors to FS decisions."""

    def decide(self, probability: float) -> Decision:
        raise NotImplementedError

    def decide_batch(self, probabilities: Sequence[float]) -> list[Decision]:
        return [self.decide(float(p)) for p in probabilities]

    def is_match(self, probability: float) -> bool:
        return self.decide(probability) is Decision.MATCH


class ThresholdClassifier(Classifier):
    """Two- or three-band threshold classifier on posterior probability.

    Parameters
    ----------
    tau:
        Match threshold: pairs with ``probability >= tau`` are matches.
    possible_low:
        Optional lower threshold for the *possible match* band.  When set,
        ``possible_low <= p < tau`` is POSSIBLE_MATCH and ``p < possible_low``
        is NON_MATCH.  When ``None`` the decision is two-band and any
        sub-threshold pair is NON_MATCH.
    """

    def __init__(
        self,
        tau: float = 0.85,
        possible_low: Optional[float] = None,
    ) -> None:
        self.tau = float(tau)
        self.possible_low = None if possible_low is None else float(possible_low)
        if self.possible_low is not None and self.possible_low > self.tau:
            raise ValueError("possible_low must be <= tau")

    def decide(self, probability: float) -> Decision:
        if probability >= self.tau:
            return Decision.MATCH
        if self.possible_low is not None and probability >= self.possible_low:
            return Decision.POSSIBLE_MATCH
        return Decision.NON_MATCH

    def __repr__(self) -> str:
        band = (
            f"possible_low={self.possible_low}"
            if self.possible_low is not None
            else "two-band"
        )
        return f"ThresholdClassifier(tau={self.tau}, {band})"


def matches_at_threshold(
    candidates: Sequence[ScoredCandidate[T]],
    tau: float,
) -> list[MatchResult[T]]:
    """Return the candidates whose posterior is at or above ``tau``.

    Results are sorted by probability descending.
    """
    out = [
        MatchResult(
            record=c.record,
            match_probability=c.probability,
            match_weight=c.match_weight,
            blocking_score=c.blocking_score,
            candidate_position=c.position,
        )
        for c in candidates
        if c.probability >= tau
    ]
    out.sort(key=lambda m: m.match_probability, reverse=True)
    return out