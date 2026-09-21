"""Capture-recapture (Lincoln-Petersen) estimation of the match prior.

A single, distribution-free cross-check of the Fellegi-Sunter base prior
``pi`` (probability two random records match).  Whereas EM's ``pi`` passes
through the comparison model -- so conditional-independence misspecification
can misstate it -- capture-recapture uses only **marginal match tallies** from
two runs, and so cannot be distorted by comparison-model error.  It does not
estimate ``m/u``; it exists to bound ``pi`` for the prior sweep.

The practitioner supplies three counts obtained from **two independent** match
runs over the same records (different field subsets, blocking schemes, or
models):

* ``n1`` -- matches found by run 1,
* ``n2`` -- matches found by run 2,
* ``m``  -- matches found by **both** (the recapture overlap).

Chapman's bias-corrected estimator then gives the total number of matches,
:math:`\\hat{N} = ((n_1+1)(n_2+1))/(m+1) - 1`, the prior is
:math:`\\hat{\\pi} = \\hat{N}/N_{\\text{pairs}}`, and Seber's variance drives a
lognormal interval on :math:`\\hat{N}` (hence on :math:`\\pi`), giving the
practitioner a defensible band to sweep.

Independence is the user's responsibility: two nested thresholds or two
runs that use the same model/data are correlated and bias the estimate (the
bookkeeping only makes sense if the captures are genuinely independent).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = ["CaptureRecapturePrior", "estimate_prior_capture_recapture"]

#: Counts at or below this overlap are too small for a trustworthy estimate.
_MIN_OVERLAP = 7


def _z_two_sided(coverage: float) -> float:
    """Two-sided normal quantile ``z`` with ``P(|Z| <= z) == coverage``.

    Inverted by Newton iteration on ``math.erf`` (no scipy dependency).
    """
    cov = float(coverage)
    if not 0.5 < cov < 1.0:
        raise ValueError("confidence must be in (0.5, 1.0)")
    z = 1.0
    inv = math.sqrt(2.0) / math.sqrt(math.pi)
    for _ in range(60):
        err = math.erf(z / math.sqrt(2.0)) - cov
        step = err / (inv * math.exp(-0.5 * z * z))
        z -= step
        if abs(step) < 1e-12:
            break
    return z


@dataclass
class CaptureRecapturePrior:
    """Lincoln-Petersen estimate of the base match prior and its interval.

    ``prior`` is :attr:`matches_estimate` / ``total_pairs`` and is clipped to
    the calibration-policy range ``(0, 0.5]``; ``prior_ci`` is the same ratio
    applied to :attr:`matches_ci`.  When ``total_pairs`` is ``None`` the prior
    fields are ``None`` and only the match-total estimates are populated.
    """

    n1: int
    n2: int
    m: int
    total_pairs: Optional[int]
    matches_estimate: float
    matches_ci: Tuple[float, float]
    prior: Optional[float]
    prior_ci: Optional[Tuple[float, float]]
    standard_error: float
    confidence: float

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        if self.prior is None:
            return (
                f"CaptureRecapturePrior(matches ~ {self.matches_estimate:.1f} "
                f"[{self.matches_ci[0]:.1f}, {self.matches_ci[1]:.1f}])"
            )
        return (
            f"CaptureRecapturePrior(prior ~ {self.prior:.3f} "
            f"[{self.prior_ci[0]:.3f}, {self.prior_ci[1]:.3f}])"
        )


def estimate_prior_capture_recapture(
    n1: int,
    n2: int,
    m: int,
    total_pairs: Optional[int] = None,
    confidence: float = 0.95,
) -> CaptureRecapturePrior:
    """Estimate the match prior (and an interval) by capture-recapture.

    Parameters
    ----------
    n1, n2:
        Matches found by the first / second **independent** run.
    m:
        Matches found by **both** runs (the recapture overlap).
    total_pairs:
        The pair domain, ``C(n_records, 2)`` for whole-file dedup or
        ``n_a * n_b`` for two-file linkage.  ``None`` returns match-count
        estimates only (no prior).
    confidence:
        Two-sided coverage of the reported interval.

    Returns
    -------
    :class:`CaptureRecapturePrior` with the Chapman point estimate
    (:attr:`matches_estimate`) and lognormal interval (:attr:`matches_ci`),
    and -- when ``total_pairs`` is given -- the prior :attr:`prior` and its
    :attr:`prior_ci` band, both clipped to ``(0, 0.5]``.

    Raises
    ------
    ValueError
        If ``m == 0`` (no overlap -> undefined), ``m > min(n1, n2)`` (an
        overlap cannot exceed a run's count), or ``confidence`` is out of
        ``(0.5, 1.0)``.

    Notes
    -----
    ``n1``/``n2``/``m`` must come from runs that are **independent** of one
    another; see the module docstring.  With ``m < 7`` the estimate is
    wide/unreliable and a :class:`UserWarning` is raised.
    """
    n1, n2, m = int(n1), int(n2), int(m)
    if n1 < 0 or n2 < 0 or m < 0:
        raise ValueError("n1, n2 and m must be non-negative")
    if m == 0:
        raise ValueError("m must be > 0: no recaptured matches means the "
                         "estimate is undefined (increase overlap / use "
                         "independent captures)")
    if m > min(n1, n2):
        raise ValueError("m cannot exceed min(n1, n2) (a run's own match count)")
    if m < _MIN_OVERLAP:
        warnings.warn(
            f"capture-recapture with a small overlap (m={m} < {_MIN_OVERLAP}) "
            "gives a wide, unreliable interval; prefer larger captures "
            "or a different estimator",
            UserWarning,
            stacklevel=2,
        )

    # Chapman's bias-corrected Lincoln-Petersen estimate.
    matches = ((n1 + 1) * (n2 + 1)) / (m + 1) - 1.0
    # Seber (1982) variance of the Chapman estimator.
    var = ((n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m)) / (
        (m + 1) ** 2 * (m + 2)
    )
    se = math.sqrt(var)

    unobserved = matches - m
    z = _z_two_sided(confidence)
    if unobserved > 1e-12 and var > 0:
        # Lognormal interval on the unobserved total ((N-m) ~ multiplicative).
        c = math.exp(z * math.sqrt(math.log1p(var / (unobserved * unobserved))))
        matches_ci = (float(unobserved / c + m), float(unobserved * c + m))
    else:
        matches_ci = (float(m), float("inf"))

    if total_pairs is None:
        return CaptureRecapturePrior(
            n1=n1, n2=n2, m=m, total_pairs=None,
            matches_estimate=float(matches), matches_ci=matches_ci,
            prior=None, prior_ci=None, standard_error=se, confidence=confidence,
        )

    total = int(total_pairs)
    if total <= 0:
        raise ValueError("total_pairs must be > 0")

    def clip_prior(x: float) -> float:
        return float(min(max(x, 1e-8), 0.5))

    return CaptureRecapturePrior(
        n1=n1, n2=n2, m=m, total_pairs=total,
        matches_estimate=float(matches), matches_ci=matches_ci,
        prior=clip_prior(matches / total),
        prior_ci=(clip_prior(matches_ci[0] / total),
                  clip_prior(matches_ci[1] / total)),
        standard_error=se, confidence=confidence,
    )
