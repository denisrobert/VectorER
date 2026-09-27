# Calibrating π and choosing the operating point

This note explains what the Fellegi-Sunter base prior `π` is actually
identifiable from, why "calibrating π" is usually the wrong question for
matching performance, and the workflow that gives you the best achievable
combination of F1 / precision / recall.

If you only read one paragraph: **for decisions, `π` and the threshold `τ` are
aliased into a single degree of freedom.** The quantity you can refine — and
should optimize — is the *operating point* on the match-weight scale,
`κ = logit(τ) − logit(π)`. `π` matters separately only when you need a
probability readout or expected counts, and then it should be reported as an
interval.

---

## 1. The alias: `π` and `τ` are one parameter

The match weight (log Bayes factor) of the Fellegi–Sunter model [1] is

```
W = Σ log(mₖ / uₖ)
```

which is **independent of `π`**. The posterior follows from the prior by

```
posterior odds = (π/(1−π)) · eᵂ
p = σ(W + logit(π))
```

The match decision `p ≥ τ` is therefore

```
W ≥ logit(τ) − logit(π) =: κ            (1)
```

so the decision depends on `(π, τ)` only through the single scalar `κ`. Every
`(π, τ)` pair with the same `κ` gives **identical** match decisions — a
one-parameter family (anti-diagonal level sets in the `(π, τ)` plane), not two
independent knobs.

With explicit decision costs the same thing happens: the optimal rule accepts
iff

```
W > log(C₁₀ / C₀₁) − logit(π)           (2)
```

so `π` and the cost ratio combine additively into one threshold. `π`, `τ` and
costs are mutually aliased for classification; only the combination is
identified.

**Why the "π is miscalibrated" problem feels real but isn't, for decisions.**
EM's prior estimate is genuinely biased (Yancey 2004 [20]; Belin & Rubin 1995 [10]), but
that bias is equivalent to an unknown shift in the decision constant. You do
not have to fix it to match well — you have to choose `κ` well. The bias only
bites when you (a) read the posterior as a probability, or (b) let EM re-fit
`m/u` jointly with `π`, because then changing `π` also moves `W` (the level
sets warp). Freezing the prior (`fixed_prior`) makes (1) exact.

### Consequence for tuning

- Tuning `π` at fixed `τ` **is** tuning `τ` at fixed `π`. Sweeping both is
  sweeping the same axis twice.
- The π-free parameterization — threshold on the match weight `W` — is the
  clean one, and is what the practitioner literature recommends.
- Keep a `π` convention only so `τ` has a meaning; it is a coordinate choice,
  not a calibration.

---

## 2. What to optimize: the metric decides the threshold

For a fixed `m/u` (hence fixed `W`), matching performance is a function of `κ`
alone. Which `κ` is best depends entirely on the objective — and the required
thresholds are *not* the same:

| objective | the `κ` that optimizes it |
|---|---|
| accuracy / 0-1 loss at equal costs | posterior `p` = 0.5 |
| cost-weighted (eq. 2) | `logit(π) + log(C₁₀/C₀₁)` relative shift |
| precision at fixed recall (or vice-versa) | read off the P-R curve |
| maximize F1 / Fβ | **generally not `p = 0.5`** |

Maximizing F1 is not the same as thresholding the posterior at 0.5, and when
positives are rare the F1-optimal threshold is typically **well below** 0.5.
F1 is a *set-level* utility — it is not a sum of per-pair F1 scores — so it has
to be optimized over the held-out pair set as a whole; for a *fixed*
classifier there is a plug-in/consistent rule giving the F1-optimal threshold
(Lipton, Elkan & Naryanaswamy 2014; Koyejo et al. 2014). Since ER matches are
rare pairs, the F1-optimal operating point is very often on the low side of
the posterior scale — which is exactly why sweeping `τ` only over
`0.5 … 0.99` can hide the best achievable F1. Threshold selection and
quality measurement in linkage practice are treated at length by
Christen (2012) [14].

**Practical rule.** Scan `κ` (equivalently: emit match weights and threshold
them) over a wide range on held-out labelled pairs, not `τ` over a narrow
high range. Then translate your chosen `κ*` back to whatever `τ` you want to
report, for your chosen `π` convention:

```
τ* = σ(κ* + logit(π))                   (3)
```

Because of (3), reporting a `(π, τ)` pair without the `κ` is under-determined
for reproducibility: state `κ` (or state both).

### 2.1 The exact curve: one pass, no grid

Because the operating point is a single scalar, the whole P-R curve of a
**fixed** scorer can be computed exactly in one pass — no `(π, τ)` grid and no
refitting:

```python
from vectorer import match_weight_curve

labelled = [(left_record, right_record, label), ...]   # label truthy = true match
curve = match_weight_curve(scorer, labelled)
best = curve.best_f1                                   # exact F1 optimum
print(best.f1, best.threshold)                         # posterior threshold tau
print(best.match_weight_bits())                        # total match weight (bits, incl. prior)
print(best.pi_free_weight(scorer.prior))               # this note's kappa = logit(tau) - logit(pi)
```

The score is a function of the comparison-level pattern and so takes **finitely
many values** on any labelled set; the curve is therefore an exact **step
function** whose breakpoints are the distinct scores attained.  `curve.points`
holds every attainable operating point (ordered by descending threshold, one
per breakpoint plus the accept-nothing endpoint), so optimising is a *sort*,
not a search — `O(n log n)` — and the best point under any of F1 / precision /
recall is read off directly (`curve.best_for("recall")`, etc.).  This is the
concrete form of the sweep in step 5 below, and it is what
`benchmarks/benchmark_lp_prior_sweep.py --eval-mode curve` (the default) uses;
`--eval-mode grid` keeps the older `fixed_prior × τ` sweep for comparison.

Note the distinction: this is a 1-D sweep of **one** curve, which requires
**fixed `m/u`**.  Sweeping `(π, τ)` with an EM refit per point explores a
*family* of curves (changing `π` changes the model), so it is not the same
thing — settle the weights first, then select the operating point on the exact
curve.

---

## 3. `κ` is not the only lever — and often not the binding one

`κ` moves you along the P-R curve; it cannot move the curve itself. The
levers that do:

1. **Candidate/blocking recall (the recall ceiling).** No threshold recovers a
   pair blocking never generated. If precision is ~1.0 and recall plateaus, the
   binding constraint is blocking (canopy parameters, `overlap_m`, HNSW
   `ef_search`, blocking rules), not `κ`.
2. **`m/u` quality — or fit the weights discriminatively.** u from large random
   pair samples (Winkler 2006 [2]; Herzog, Scheuren & Winkler 2007 [24]); m from
   EM on match-enriched blocks (Jaro 1989 [6]; Yancey 2004 [20]); supervised
   `m/u` from clerically reviewed pairs is the strongest option when available.
   Where labelled pairs exist, the score weights can instead be fit
   **discriminatively**: a logistic regression over the same comparison-level
   indicators, trained to maximise a smoothed expected F-measure rather than
   the likelihood (Jansche 2005 [25]). This is the FS score form in disguise —
   per-level coefficients ↔ `log(m/u)` and the intercept ↔ `logit(π)` — so it
   drops into the existing scorer, and the framework already builds the needed
   indicator matrix (`_assign_levels` inside `calibrate_from_pairs` / `fit_em`).
   Prefer it when conditionally dependent comparisons bias the generative `m/u`
   (the §7 caveat) or when the metric itself is the training objective; it
   costs you the interpretability of `m/u` and `π`, and it still needs labels.
   Crucially, the recall/precision balance is set by the training objective
   (`β`), so it shapes the fitted **weights**, not merely the threshold: a
   fixed score can only be moved *along* its P-R curve by `κ`, whereas
   metric-driven training can reshape the curve. That is the concrete way this
   route can beat generative `m/u` even with labels — and only under
   misspecification or limited data, since a correctly specified model's
   F-optimal rule is itself just a threshold. The price is that the preference
   is baked into the model, so changing the operating point means retraining,
   where FS keeps `π`/weights and the `κ` threshold separable. `m/u` are not the
   only way to obtain the weights — they are the generative way.
3. **Comparison levels and thresholds.** The level boundaries set `W`'s
   resolution; poor boundaries cap achievable separation regardless of `κ`.
4. **Term-frequency / population adjustments** (`base_records`, TF weight
   tables) — these change `u` for common values and so change `W`.
5. **Dedup clustering stage.** In batch dedup, `τ` gates the edges that the
   Swoosh closure merges; the merge function (union vs representative) and
   closure semantics affect cluster-level P/R beyond pair-level `κ`.
6. **Block-level heterogeneity (do *not* use per-block `κ`).** Blocks differ in
   match prevalence, but deployment applies **one** threshold, so the target is
   a single `κ` optimized on the **pooled** candidate pairs (pooling already
   weights blocks by their pair counts and prevalences). Averaging per-block
   optima is invalid — thresholds are not additive, and the mean is not the
   pooled optimum. If the block key genuinely carries prior information (rare-
   name blocks have higher match rates), **fold it into the score** as an extra
   comparison so `m/u` give it an additive log-Bayes-factor (a per-block prior
   offset inside `W`); then one global `κ` is again optimal. Per-block metrics
   are diagnostic, not a basis for thresholding.

Diagnose in this order: (a) candidate recall (can the true pairs even be
scored?), (b) separability (`m/u` — is the P-R curve good at *its* best
point?), (c) operating point (`κ`).

---

## 4. When you actually need `π` (not just `κ`)

`π` is separately meaningful when you need:

- a **probability readout** ("this pair has a 0.98 chance of being a match"),
- **expected counts** ("about N true matches in this file"),
- **portability** across datasets or blocks, where the pair density differs,
- the **EM coupling** — if `π` is estimated jointly, it affects `m/u`.

For decisions and for F1/precision/recall tuning, none of the above applies:
use `κ`.

When you do need `π`, the defensible options and their standing:

| method | standing / caveat |
|---|---|
| EM mixture weight | standard but known biased (usually **upward**); Jaro 1989 [6]; Yancey 2004 [20]; Belin & Rubin 1995 [10] |
| labelled/audited pair sample | unbiased if the sample is random and large enough; hopeless for very rare matches at full-pair scale |
| blocking-corrected EM | necessary because the EM prior is a *blocked* rate; divide by blocking recall |
| dual-system / capture-recapture | needs genuinely independent captures; false positives and correlated captures bias it (Ding & Fienberg 1994) |
| Bayesian hierarchical ER | returns a posterior *distribution* for the match rate — the honest object (Tancredi & Liseo 2011; Gutman, Afendulis & Zaslavsky 2013; Steorts, Hall & Fienberg 2016) |
| post-hoc calibration of `p` | recalibrate the returned posterior against a labelled sample (Platt 1999; Niculescu-Mizil & Caruana 2005) |

**Never publish a single `π` as ground truth.** Report an interval, and state
the operating point as `κ` so the decision is reproducible whatever `π`
convention you chose.

---

## 5. Recommended workflow for best F1 / precision / recall

1. **Define the objective quantitatively.** "Maximise F1", or "maximise recall
   subject to precision ≥ 0.99", or a cost ratio. F1 weights precision and
   recall equally; if your costs are asymmetric, use (2)/P-R-with-constraint
   instead. Write it down before tuning.
2. **Build a held-out labelled eval set — from the deployment decision domain.**
   Positives from a known registry / audited sample; negatives that are *genuine*
   non-matches.  Sample it **uniformly from the same pair domain the threshold
   governs** and with matching class prevalence: if the scorer only ever sees
   blocked candidates (the usual batch-ER case), sample the *candidate* pairs —
   not the enriched training blocks, and not the full pair space (whose ≈10⁻⁷
   match density is both the wrong prevalence and far too rare to label).  The
   F1-optimal `κ` depends on the positive rate, so a set from a different domain
   yields the wrong operating point unless it is **reweighted**.  Blocking
   **recall** is a separate quantity — a candidate-domain sample cannot see the
   pairs blocking never generated — so estimate it separately (the `recall`
   factor) rather than by widening the eval set.  Sample **pairs uniformly**
   within the chosen domain, not whole blocks: a block that contains no
   positives simply contributes negatives — usually the *hard* look-alike
   negatives that set `κ` — so dropping such blocks would bias `κ` permissive.
   (If you must sample block-wise, weight by block size so the pooled
   prevalence is preserved, and compute precision/recall **pooled**, never per
   block.)  Tune and report on held-out data only.
3. **Fit `m/u` well.** u from a large random pair sample; m from EM on
   match-enriched blocks; supervised m/u if clerically-reviewed pairs exist.
   **Freeze `π`** (any convention, e.g. `1e-4`) so `(π, τ)` stays on a known
   level set.
4. **Fit and calibrate on *disjoint* data (or cross-fit).** The optimal `κ` is a
   function of the model, so choosing it on the pairs that fitted the weights
   overfits the score distribution — Koyejo's two-step estimator (Alg 1) fits
   `η̂` on one split `S1` and picks the threshold on a disjoint `S2`, which is
   what its consistency guarantee rests on.  Weights fitted **supervised**
   (`calibrate_from_pairs`, or the discriminative `β`-training of §3) must not
   share labelled pairs with the `κ`-calibration set; pure EM sees no labels, so
   that leak is absent there, but a clean split still keeps the calibration set
   a genuine holdout.  Prefer **k-fold cross-fitting** to a single split: fit on
   `K−1` folds, tune `κ` on the held-out fold's scores, rotate, aggregate — no
   leakage, and all data used for both stages.
5. **Sweep `κ`, not `τ` — or, better, read the exact curve.** With `m/u`
   frozen, build the full P-R/F1 curve of the fixed scorer in one pass with
   `match_weight_curve(scorer, labelled)` (§2.1): its `points` are every
   attainable operating point, so no grid is needed. If you prefer the explicit
   sweep, scan `κ` (the match-weight threshold) rather than `τ` over a narrow
   high range, and include the low end — the F1 optimum is often below the
   `τ = 0.5` point.
6. **Pick `κ*` by the objective** (`curve.best_for("f1")`, or a
   precision-constrained point), then derive the reported `τ* = σ(κ* +
   logit(π))`.
7. **Check the recall ceiling.** If precision ≈ 1 and recall stalls, revisit
   blocking/candidate recall before touching `κ` again.
8. **Report honestly.** Give the chosen `κ`/`τ`, the eval protocol, and — if a
   probability scale is required — a `π` interval with its source. Cite the
   P-R curve, not a single number, when the objective is close between
   operating points.

The framework's `fit_em(fixed_prior=...)` + threshold sweep is exactly step 5,
and the match-weight threshold is the lever practitioner tooling exposes
(Splink [19]); the `benchmark_lp_prior_sweep.py` capture-recapture band is a
*bounding* input for step 8, not a substitute for step 5.

---

## 6. Worked example (from `benchmark_lp_prior_sweep.py`)

On the 322k demo population the LP band put the full-file prior at
`π ≈ 9×10⁻⁸` while the framework default is `π = 10⁻⁴`; both were scored at
`τ = 0.5`. In `κ` terms (eq. 1), `logit(0.5) = 0`:

| config | `π` | `τ` | `κ` (nats) | best F1 |
|---|---|---|---|---|
| default prior | 1e-4 | 0.5 | 0 + 9.21 = **9.21** | 0.9998 |
| LP band | 9e-8 | 0.5 | 0 + 16.22 = **16.22** | 0.8302 |

The two "different priors" are really two `κ` values ~7 nats (≈10 bits of
match weight) apart — the high-`κ` config demands far stronger evidence per
match, and that alone explains the recall drop (0.9995 → 0.71) at identical
`τ`. Within the LP band the sweep rows were flat across priors, the signature
of moving *along* a level set. The actionable reading: sweep `κ` directly, and
extend below `τ = 0.5` — the grid's `τ ≥ 0.5` floor, not `π`, bounded the
achievable F1.

---

## References

Numbering follows the repository's **global** sequence: [1]–[20] are listed in
[`.docs/architecture.md`](architecture.md) §9, and the new numbers continue from
there. PDFs and study summaries are curated in `.source-papers/` with an
availability manifest in its `README.md`.

- [10] Belin, T. R., & Rubin, D. B. (1995). A method for calibrating false-match rates in record linkage. *JASA* 90(430), 694–707. [DOI 10.1080/01621459.1995.10476563](https://doi.org/10.1080/01621459.1995.10476563). Study notes: `.source-papers/10_belin_rubin_1995/`.
- [14] Christen, P. (2012). *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection.* Springer. [DOI 10.1007/978-3-642-31164-2](https://doi.org/10.1007/978-3-642-31164-2). (Threshold selection and linkage quality measures; owned by the maintainer as a physical copy.)
- [22] Ding, Y., & Fienberg, S. E. (1994). Dual system estimation of census undercount in the presence of matching error. *Survey Methodology* 20(2), 149–158. `.source-papers/22_ding_dual_system_estimation_1994.pdf`.
- [1] Fellegi, I. P., & Sunter, A. B. (1969). A theory for record linkage. *JASA* 64(328), 1183–1210. [DOI 10.1080/01621459.1969.10501049](https://doi.org/10.1080/01621459.1969.10501049). (The decision rule with prior odds and error costs; `.source-papers/01_fellegi_sunter_1969.pdf`.)
- [23] Gutman, R., Afendulis, C. C., & Zaslavsky, A. M. (2013). A Bayesian procedure for file linking to analyze end-of-life medical costs. *JASA* 108(501), 34–47. [PMC3640583](https://pmc.ncbi.nlm.nih.gov/articles/PMC3640583/) · `.source-papers/23_gutman_bayesian_2013.pdf`.
- [24] Herzog, T. N., Scheuren, F. J., & Winkler, W. E. (2007). *Data Quality and Record Linkage Techniques.* Springer. [DOI 10.1007/0-387-69505-2](https://doi.org/10.1007/0-387-69505-2). (Practitioner reference on `m/u` estimation and linkage operations; owned by the maintainer as a physical copy.)
- [25] Jansche, M. (2005). Maximum expected F-measure training of logistic regression models. *Proceedings of HLT–EMNLP 2005*, 692–699. [ACL Anthology H05-1087](https://aclanthology.org/H05-1087/). (Discriminative, metric-driven training of a logistic classifier; used here as the alternative to generative `m/u` estimation over the FS comparison-level indicators.)
- [6] Jaro, M. A. (1989). Advances in record-linkage methodology as applied to matching the 1985 Census of Tampa, Florida. *JASA* 84(406), 414–420. [DOI 10.1080/01621459.1989.10478785](https://doi.org/10.1080/01621459.1989.10478785). Study summary: `.source-papers/06_jaro_1989/JaroSummary.md`. (EM estimation of `m/u` in practice; the Jaro string comparator.)
- [26] Koyejo, O., Natarajan, N., Ravikumar, P., & Dhillon, I. S. (2014). Consistent binary classification with generalized performance metrics. *Advances in Neural Information Processing Systems 27.* [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2014/hash/98053046e0dce5c7d946c67b96a85e18-Abstract.html).
- [19] Linacre, R., et al. (2022). Splink: Free software for probabilistic record linkage at scale. *IJPDS* 7(3). [DOI 10.23889/ijpds.v7i3.1794](https://doi.org/10.23889/ijpds.v7i3.1794) · documentation [moj-analytical-services.github.io/splink](https://moj-analytical-services.github.io/splink/). (Practical guidance on the prior and match-weight thresholding.)
- [28] Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014). Thresholding classifiers to maximize F1 score. [arXiv:1402.1892](https://arxiv.org/abs/1402.1892).
- [29] Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML.* [DOI 10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430).
- [30] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press. (Post-hoc probability calibration.)
- [32] Steorts, R. C., Hall, R., & Fienberg, S. E. (2016). A Bayesian approach to graphical record linkage and deduplication. *JASA* 111(516), 1660–1672. [arXiv:1312.4645](https://arxiv.org/abs/1312.4645).
- [33] Tancredi, A., & Liseo, B. (2011). A hierarchical Bayesian approach to record linkage and population size problems. *Annals of Applied Statistics* 5(2B), 1553–1585. [arXiv:1011.2649](https://arxiv.org/abs/1011.2649).
- [2] Winkler, W. E. (2006). *Overview of record linkage and current research directions.* U.S. Census Bureau Research Report RRS2006/02. [Link](https://www.census.gov/library/working-papers/2006/adrm/rrs2006-02.html). (u from random pairs; m from EM; practical estimation.)
- [20] Yancey, W. E. (2004). *Improving EM algorithm estimates for record linkage parameters.* U.S. Census Bureau Research Report RRS2004-01. [PDF](https://www.census.gov/content/dam/Census/library/working-papers/2004/adrm/rrs2004-01.pdf). (EM prior bias.)

The empirical kappa-alias and threshold examples above come from
`benchmarks/benchmark_lp_prior_sweep.py`; see
[`.docs/user_guide.md`](user_guide.md) §6.3 for the runnable form.