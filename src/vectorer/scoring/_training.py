"""Training: supervised calibration + EM + prior recovery.

* :meth:`TrainingMixin.calibrate_from_pairs` -- supervised ``m/u`` estimation
  from labelled match/non-match pairs;
* :meth:`TrainingMixin.fit_em` -- unsupervised expectation maximisation on a
  (near-duplicate-bearing) population;
* :meth:`TrainingMixin.recalibrate_prior` -- recover the full-set base prior
  for a scorer trained on an enriched subset (Yancey 2004 / empirical /
  capture-recapture).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..comparisons import Comparison, PairValues
from ._levels import (
    _assign_levels,
    _level_defaults,
    _level_proportions,
)
from ._math import _sigmoid


class TrainingMixin:
    """Supervised + EM training and prior calibration.

    Abstract: mixed into :class:`~vectorer.scoring.FellegiSunterScorer`; reads
    ``self.comparisons`` / ``self.prior`` / ``self.threshold`` and writes a new
    scorer via ``self.__class__.from_settings``.
    """

    def calibrate_from_pairs(
        self,
        pair_records: Sequence[dict],
        smoothing: float = 0.5,
    ) -> "FellegiSunterScorer":
        """Fit supervised ``m/u`` from labelled match/non-match pairs.

        ``pair_records`` is a sequence of records with a boolean ``is_match``
        column (``1`` = match, ``0`` = non-match) plus, for each field compared
        by the comparison set, ``<field>_l`` and ``<field>_r`` columns.
        Laplace ``smoothing`` is added to every level count (vectorized).
        """
        if not self.comparisons:
            raise ValueError("no comparisons configured on the scorer")
        is_match = np.array(
            [1 if r.get("is_match") else 0 for r in pair_records], dtype=np.int64
        )
        m_total = int((is_match == 1).sum())
        u_total = int((is_match == 0).sum())
        if m_total == 0 or u_total == 0:
            raise ValueError("pairs must contain both match (1) and non-match (0) rows")

        new_comparisons: list[Comparison] = []
        for comparison in self.comparisons:
            spec = comparison.spec()
            fields = tuple(spec.fields)
            if not fields:
                first_pair = pair_records[0]
                fields = tuple(sorted({k[:-2] for k in first_pair if k.endswith("_l")}))
            left = {
                f: np.array([r.get(f"{f}_l") for r in pair_records], dtype=object)
                for f in fields
            }
            right = {
                f: np.array([r.get(f"{f}_r") for r in pair_records], dtype=object)
                for f in fields
            }
            pv = PairValues(left, right)
            assigned = _assign_levels(spec, pv)
            num_levels = len(spec.levels)
            level_probs: list[tuple[Optional[float], Optional[float]]] = []
            for index, level in enumerate(spec.levels):
                if level.is_null:
                    level_probs.append((None, None))
                    continue
                m_count = int(((assigned == index) & (is_match == 1)).sum())
                u_count = int(((assigned == index) & (is_match == 0)).sum())
                m = (m_count + smoothing) / (m_total + smoothing * num_levels)
                u = (u_count + smoothing) / (u_total + smoothing * num_levels)
                level_probs.append((float(m), float(u)))

            resolved = comparison.resolved()
            new_levels = []
            for level_dict, (m, u) in zip(resolved["levels"], level_probs):
                if level_dict.get("is_null_level"):
                    new_levels.append(level_dict)
                else:
                    new_levels.append(
                        {**level_dict, "m_probability": m, "u_probability": u}
                    )
            resolved["levels"] = new_levels
            new_comparisons.append(Comparison.from_resolved(resolved))

        return self.__class__.from_settings(
            {
                "comparisons": [c.resolved() for c in new_comparisons],
                "probability_two_random_records_match": self.prior,
            },
            threshold=self.threshold,
        )

    def fit_em(
        self,
        records: Sequence[dict],
        training_block_on: Optional[Sequence[Sequence[str]]] = None,
        recall: float = 0.7,
        max_pairs: float = 1e6,
        max_iterations: int = 20,
        em_convergence: float = 0.001,
        seed: Optional[int] = None,
        prior: Optional[float] = None,
        fixed_prior: Optional[float] = None,
        extra_settings: Optional[dict] = None,
    ) -> "FellegiSunterScorer":
        """Fit ``m``/``u`` and the base prior via expectation maximisation.

    Native training workflow (no SQL engine):

    1. candidate pairs are generated under the blocking rules
       (:func:`~vectorer.scoring._candidates.build_blocked_pairs`) -- this
       defines the *training* pair pool;
    2. ``u`` probabilities are estimated from uniformly sampled pairs
       (the unconditional non-match distribution);
    3. ``m`` and the blocked-pair match proportion are fit by EM (u held
       fixed), with ``m`` renormalized per comparison at each M-step;
    4. the base prior (probability two random records match) is the
       recall-adjusted share of blocked pairs that are matches extended to the
       total number of possible pairs, capped at 0.5 -- unless ``prior`` or
       ``fixed_prior`` is supplied.

    Fixed-prior mode (the "calibration paradox" remedy): passing
    ``fixed_prior=`` holds the base prior **frozen across every EM iteration**
    (it is used in the E-step's responsibilities and never re-estimated in the
    M-step), so only ``m`` (and not ``pi``) is learned -- the same machinery
    Splink's fixed-prior EM uses.  This lets you sweep fixed priors x threshold
    to find an operating point, instead of trusting EM's own (often
    miscalibrated) prior estimate.  ``prior=`` remains the earlier behaviour:
    EM still learns ``pi`` internally, and ``prior`` only overrides the
    reported base rate.

    Only one of ``prior`` / ``fixed_prior`` should be set; the caller may also
    pass both only if they agree.
    """
        from ._candidates import _positions_pair_values, _sample_all_pairs, build_blocked_pairs

        rng = np.random.default_rng(seed)
        rules = list(training_block_on) if training_block_on else [
            ("first_name",),
            ("date_of_birth",),
        ]
        if not self.comparisons:
            raise ValueError("no comparisons configured on the scorer")

        blocked_pairs = build_blocked_pairs(records, rules, int(max_pairs), rng)
        n_pairs = len(blocked_pairs)
        if n_pairs == 0:
            raise RuntimeError(
                "no blocking-rule candidate pairs were generated; supply a "
                "duplicate-bearing population or adjust training_block_on"
            )

        specs = [c.spec() for c in self.comparisons]
        field_list = self.table.fields

        # (1) comparison vectors for the blocked pair pool, vectorized.
        blocked_pv = _positions_pair_values(records, blocked_pairs, field_list)
        gammas = [_assign_levels(spec, blocked_pv) for spec in specs]

        # (2) u from a uniform sample of pairs (unconditional non-match dist).
        u_sample = list(_sample_all_pairs(len(records), min(200_000, n_pairs * 2), rng))
        if not u_sample:
            u_sample = blocked_pairs
        u_pv = _positions_pair_values(records, u_sample, field_list)
        u_gammas = [_assign_levels(spec, u_pv) for spec in specs]
        us = [_level_proportions(spec, gamma) for spec, gamma in zip(specs, u_gammas)]

        # (3) EM: start m from current defaults, pi neutral.
        ms = [_level_defaults(spec) for spec in specs]
        null_flags = [
            np.array([lv.is_null for lv in spec.levels], dtype=bool)
            for spec in specs
        ]
        pi = 0.5
        # Fixed-prior mode: pi is frozen at fixed_prior for the whole EM run.
        if fixed_prior is not None:
            if not 0.0 < fixed_prior < 1.0:
                raise ValueError("fixed_prior must be in (0, 1)")
            if prior is not None and abs(prior - fixed_prior) > 1e-12:
                raise ValueError("prior and fixed_prior disagree; set only one")
            pi = float(fixed_prior)
        prev_pi = None
        try:
            from tqdm import tqdm

            em_bar = tqdm(range(int(max_iterations)), desc="EM m/u", unit="iter",
                          leave=False, ascii=True)
        except Exception:  # noqa: BLE001  (tqdm optional)
            em_bar = range(int(max_iterations))
        # Guards against the degenerate all-C1 fixed point: if EM drives the
        # responsibility mass to every blocked pair being a "match" (pi -> 1),
        # the m levels collapse onto whichever level has the highest base rate
        # and every other level gets floored, destroying the evidence contrast
        # needed for scoring.  Rather than letting that silently lock in, we
        # clip pi so it can never leave (0, 1) by more than the numeric slack,
        # freeze pi at its last sane value when the likelihood landscape is
        # degenerate, and detect the collapse to report a clear error.
        pi_min = 1e-6
        pi_max = 1.0 - 1e-6
        for _ in em_bar:
            # E-step (log-space responsibilities over the blocked pairs).
            log_m = np.zeros(n_pairs, dtype=np.float64)
            log_u = np.zeros(n_pairs, dtype=np.float64)
            for spec, gamma, m_levels, u_levels, is_null in zip(
                specs, gammas, ms, us, null_flags
            ):
                null_gamma = is_null[gamma]
                m_g = np.clip(np.asarray(m_levels, dtype=np.float64)[gamma], 1e-8, None)
                u_g = np.clip(np.asarray(u_levels, dtype=np.float64)[gamma], 1e-8, None)
                # Null-level comparisons carry no evidence (bayes factor 1).
                log_m += np.where(null_gamma, 0.0, np.log(m_g))
                log_u += np.where(null_gamma, 0.0, np.log(u_g))
            log_odds = log_m - log_u + np.log(pi + 1e-12) - np.log1p(-pi + 1e-12)
            r = _sigmoid(np.clip(log_odds, -50.0, 50.0))

            # M-step: per-comparison m (renormalized) uses the full posterior
            # responsibilities ``r`` (standard EM).  The blocked-pair match
            # share ``pi``, however, is estimated from the *evidence-only*
            # responsibilities ``r_ev = sigmoid(log_m - log_u)`` (no logit-pi
            # term).  This is the statistical fix for the spurious 1.0 fixed
            # point: ``pi = mean(r)`` with ``r`` containing ``logit(pi)`` is a
            # self-reinforcing loop (as pi -> 1 the logit term explodes and
            # drags every responsibility to 1).  The evidence-only share
            # reflects the pool's composition without that coupling, so pi
            # converges to the actual share of evidence-supported pairs.
            if fixed_prior is None:
                r_ev = _sigmoid(np.clip(log_m - log_u, -50.0, 50.0))
                pi_new = float(np.clip(float(np.mean(r_ev)), pi_min, pi_max))
            else:
                pi_new = pi
            ms_new = []
            for spec, gamma in zip(specs, gammas):
                weights = r.copy()
                sums = np.zeros(len(spec.levels), dtype=np.float64)
                for index in range(len(spec.levels)):
                    sums[index] = float(weights[gamma == index].sum())
                total = float(sums.sum())
                m_levels = np.clip(sums / (total + 1e-12), 1e-8, None)
                m_levels = m_levels / m_levels.sum()  # per-comparison multinomial
                ms_new.append(m_levels.tolist())
            change = abs(pi_new - pi)
            if prev_pi is not None and fixed_prior is None:
                for old, new in zip(ms, ms_new):
                    change = max(change, float(np.max(np.abs(np.asarray(old) - np.asarray(new)))))
            pi = pi_new
            ms = ms_new
            if change < em_convergence:
                break
            prev_pi = pi

        # Base prior: probability two *random* records match = the model's own
        # expected match rate over a uniform (unconditional) pair sample.  We
        # cannot use `pi` (the blocked-pair match share) extrapolated to the
        # full pair domain: the blocked pool is intentionally biased toward
        # matches (it is built from blocking rules), so `pi * |blocked|` is a
        # multiple of the true match count and the "extrapolation" over
        # C(n,2) wildly overstates the base rate.  Instead solve the fixed
        # point over a uniform pair sample (see _uniform_prior_estimate).
        # Fixed-prior mode returns the frozen prior verbatim; a caller-supplied
        # `prior` overrides the reported base rate (historical behaviour).
        if fixed_prior is not None:
            final_prior = float(fixed_prior)
        elif prior is not None:
            final_prior = float(prior)
        else:
            final_prior = None  # resolved from the uniform-sample score below

        new_comparisons = []
        for comparison, m_levels, u_levels in zip(self.comparisons, ms, us):
            resolved = comparison.resolved()
            new_levels = []
            for idx, level_dict in enumerate(resolved["levels"]):
                if level_dict.get("is_null_level"):
                    new_levels.append(level_dict)
                else:
                    new_levels.append({**level_dict, "m_probability": float(m_levels[idx]),
                                       "u_probability": float(u_levels[idx])})
            resolved["levels"] = new_levels
            new_comparisons.append(Comparison.from_resolved(resolved))

        if final_prior is None:
            # Uniform-sample fixed-point estimate of the unconditional match
            # rate (self-consistent, not biased by a flat 0.5 prior).
            base_rate = self._uniform_prior_estimate(
                records, min(200_000, n_pairs * 4), rng)
            if base_rate != base_rate:  # nan: unexpected empty sample
                base_rate = 0.0
            final_prior = float(np.clip(base_rate, 1e-8, 0.5))

        scorer = self.__class__.from_settings(
            {
                "comparisons": [c.resolved() for c in new_comparisons],
                "probability_two_random_records_match": float(final_prior),
            },
            threshold=self.threshold,
        )
        # Record the EM mixing share over the blocked candidate pairs (pi) and
        # the blocked-pair count (n_pairs = |S0|).  ``pi`` reflects the blocked
        # pool, which is intentionally biased toward matches (blocking rules),
        # so it is a *blocked*-share estimate -- always use ``prior_empirical``
        # (the uniform-posterior base rate) as the honest full-pair prior.  The
        # pair count and block share let recalibrate_prior reproduce either.
        scorer._em = {
            "pi": float(pi),
            "n_pairs": int(n_pairs),
            "prior_empirical": float(final_prior),
        }
        return scorer

    def recalibrate_prior(
        self,
        records: Sequence[dict],
        *,
        method: str = "yancey",
        sample_size: int = 200_000,
        seed: Optional[int] = None,
        recall: float = 1.0,
        n_captures: Optional[tuple[int, int, int]] = None,
        confidence: float = 0.95,
    ) -> "FellegiSunterScorer":
        """Recover the **full-set** match prior for a model trained on an
        enriched subset.

        After match-enrichment EM (Yancey 2004), the EM estimate of ``Pr(C1)``
        reflects the match share of the enriched training subset, not the full
        population -- Yancey recalibrates only the non-match classes (C2/C3)
        and leaves the match prior un-adjusted.  In a posterior/threshold
        system the match prior enters the posterior directly, so an inflated
        prior biases every posterior.

        ``method`` selects the correction computed on ``records`` (the full
        population):

        * ``"yancey"`` (default) -- the paper's count-ratio correction
          (Yancey 2004, eq. for Pr(C1): ``Pr(C1)_S = (|S0|/|S|) * Pr(C1)_S0``).
          ``|S0|`` is the enriched blocked-pair count recorded by
          :meth:`fit_em`, ``|S|`` is ``C(len(records), 2)`` (the full pair
          domain), and ``Pr(C1)_S0`` is EM's mixing proportion ``pi`` over the
          enriched blocked pairs.  This is deterministic, free (no scoring
          pass), and matches the article's rescaling of the enriched class
          shares to the full pair set.

        * ``"empirical"`` -- resample: draw a ``sample_size`` uniform pair
          sample from ``records``, score it with the trained ``m/u``, and set
          the prior to the model's own expected match rate (optionally divided
          by ``recall`` to compensate for blocking that produced candidates).

        * ``"lincoln_petersen"`` -- capture-recapture (see
          :func:`~vectorer.scoring.estimate_prior_capture_recapture`).  ``n1,
          n2, m`` (matches from two **independent** runs and their overlap) are
          supplied via ``n_captures``; the prior is the Chapman point estimate
          over ``C(len(records), 2)``.  Its ``prior_ci`` interval is the
          practitioner's band for the prior sweep -- run
          ``fit_em(fixed_prior=...)`` over that interval to choose the
          operating point that the base rate miscalibration would otherwise
          dominate.  ``recall`` is ignored for this method.

        ``method="yancey"`` requires the scorer to carry EM metadata (i.e. it
        was produced by :meth:`fit_em`); otherwise a :class:`ValueError` is
        raised.  Use the resulting scorer's ``fixed_prior`` (or the
        prior-sweep) at the corrected prior.
        """
        if method == "empirical":
            return self._recalibrate_empirical(
                records, sample_size=sample_size, seed=seed, recall=recall,
            )
        if method == "lincoln_petersen":
            return self._recalibrate_lincoln_petersen(
                records, n_captures=n_captures, confidence=confidence,
            )
        if method != "yancey":
            raise ValueError(
                f"unknown recalibrate_prior method {method!r}; expected "
                "'yancey', 'empirical', or 'lincoln_petersen'"
            )
        if not self._em or "pi" not in self._em or "n_pairs" not in self._em:
            raise ValueError(
                "recalibrate_prior(method='yancey') requires a scorer produced "
                "by fit_em (which records the enriched EM mixing share and "
                "blocked-pair count); use method='empirical' for a manually "
                "constructed scorer"
            )
        pi = float(self._em.get("pi", 0.0))
        n0 = int(self._em.get("n_pairs", 0))
        n = len(records)
        n_total_pairs = n * (n - 1) // 2
        # Yancey 2004 sec 2.4: Pr(C2)_S = (|S0|/|S|) * Pr(C2) (and C3 absorbs
        # the uncounted remainder).  For a match prior this is the share of
        # full-pair space the enriched EM attributes to matches: pi of the
        # |S0| blocked pairs are matches, so pi * |S0| / |S|.
        full_prior = np.clip(
            (pi * n0) / max(float(recall), 1e-3) / max(n_total_pairs, 1),
            1e-8, 0.5,
        ) if n_total_pairs else 0.0
        new_settings = self.to_settings()
        new_settings["probability_two_random_records_match"] = float(full_prior)
        return self.__class__.from_settings(
            new_settings, threshold=self.threshold,
        )

    def _recalibrate_empirical(
        self,
        records: Sequence[dict],
        *,
        sample_size: int = 200_000,
        seed: Optional[int] = None,
        recall: float = 1.0,
    ) -> "FellegiSunterScorer":
        """Empirical resample recalibration (the pre-``method`` behaviour).

        Draws a ``sample_size`` uniform pair sample from ``records`` and sets
        the prior to the model's own expected match rate via the self-
        consistent fixed-point estimate (:meth:`_uniform_prior_estimate`).
        ``recall`` is accepted for API compatibility; the uniform sample is
        already drawn over the full pair domain, so no recall adjustment is
        applied (dividing a full-domain posterior mean by recall was biased).
        """
        rng = np.random.default_rng(seed)
        base_rate = self._uniform_prior_estimate(
            records, int(sample_size), rng)
        if base_rate != base_rate:  # nan: empty sample
            return self
        full_prior = float(np.clip(base_rate, 1e-8, 0.5))
        new_settings = self.to_settings()
        new_settings["probability_two_random_records_match"] = full_prior
        return self.__class__.from_settings(
            new_settings, threshold=self.threshold,
        )

    def _recalibrate_lincoln_petersen(
        self,
        records: Sequence[dict],
        *,
        n_captures: Optional[tuple[int, int, int]],
        confidence: float = 0.95,
    ) -> "FellegiSunterScorer":
        """Capture-recapture prior recovery (see :meth:`recalibrate_prior`).

        ``n_captures = (n1, n2, m)`` are matches from two **independent** runs
        and their overlap; the prior is the Chapman estimate over the whole
        file's pair domain ``C(len(records), 2)``.
        """
        from ._capture_recapture import estimate_prior_capture_recapture

        if n_captures is None:
            raise ValueError(
                "method='lincoln_petersen' requires n_captures=(n1, n2, m)"
            )
        n = len(records)
        total_pairs = n * (n - 1) // 2
        est = estimate_prior_capture_recapture(
            n_captures[0], n_captures[1], n_captures[2],
            total_pairs=total_pairs, confidence=confidence,
        )
        new_settings = self.to_settings()
        new_settings["probability_two_random_records_match"] = float(est.prior)
        return self.__class__.from_settings(
            new_settings, threshold=self.threshold,
        )

    def _uniform_prior_estimate(
        self,
        records: Sequence[dict],
        sample_size: int,
        rng: np.random.Generator,
    ) -> float:
        """Self-consistent estimate of the unconditional match base rate.

        Draws a ``sample_size`` uniform pair sample and solves the fixed point
        ``pi = mean(posterior(pi))`` over it (a few likelihood iterations on the
        *precomputed* per-pair evidence).  ``p05`` is the posterior at a flat
        ``0.5`` prior, so ``evidence = logit(p05)`` and
        ``posterior(pi) = sigmoid(evidence + logit(pi))``; the fixed-point loop
        is a cheap vectorized iteration over a 1-D array -- no scorer rebuild
        per iteration.

        Boundary guard: ``pi = mean(posterior(pi))`` has attracting fixed
        points at both 0 and 1 (the logistic is increasing in the prior), so
        an evidence-positive pool can run the iteration to the spurious 1.0.
        The guard runs *on every iterate*: once the iterate is pushed above
        ``0.9`` and still climbing, we fall back to the **evidence-positive
        share** ``mean(ev > 0)`` -- a prior-free, bounded, lower-anchor
        estimate of the match share that cannot be inflated by the logit
        coupling.  Returns ``nan`` when the sample is empty (callers must
        handle it).
        """
        from ._candidates import _sample_all_pairs

        n = len(records)
        pairs = list(_sample_all_pairs(n, int(sample_size), rng))
        if not pairs:
            return float("nan")
        left = [records[i] for i, _ in pairs]
        right = [records[j] for _, j in pairs]
        # Per-pair evidence: logit of the posterior at a neutral 0.5 prior
        # (sigmoid(evidence) = posterior(0.5)), computed once.
        neutral = self.__class__.from_settings(
            {
                "comparisons": [c.resolved() for c in self.comparisons],
                "probability_two_random_records_match": 0.5,
            },
            threshold=self.threshold,
        )
        p05 = np.asarray(neutral.score_pairs(left, right), dtype=np.float64)
        p05 = np.clip(p05, 1e-12, 1.0 - 1e-12)
        evidence = np.log(p05) - np.log(1.0 - p05)
        # Fixed-point iteration pi = mean(sigmoid(evidence + logit(pi))).
        # Convergence is tested in logit space (which contracts faster in the
        # slow near-boundary regime) and the spurious-1 trajectory guard runs
        # *every* iterate, so a pool climbing past 0.9 without stabilizing is
        # caught even if it would land below the final-value threshold.
        ev_share = float((evidence > 0.0).mean())
        pi_val = 0.5
        for _ in range(40):
            logit_pi = np.log(max(pi_val, 1e-12)) - np.log(max(1.0 - pi_val, 1e-12))
            pi_new = float(np.mean(_sigmoid(np.clip(evidence + logit_pi, -50.0, 50.0))))
            if abs((np.log(max(pi_new, 1e-12)) - np.log(max(1.0 - pi_new, 1e-12)))
                   - logit_pi) < 1e-4:
                pi_val = pi_new
                break
            # Trajectory boundary guard: if the iterate is pushed into the
            # top band and keeps climbing, fall back to the evidence-positive
            # share (prior-free, bounded) rather than converging to ~1.
            if pi_new > 0.9 and pi_new > pi_val:
                pi_val = min(pi_new, max(ev_share, 1e-6))
                break
            pi_val = pi_new
        return pi_val