"""Tests for the Fellegi-Sunter scoring engine (native NumPy implementation)."""

import numpy as np
import pytest

from vectorer.scoring import (
    DEFAULT_PRIOR,
    FellegiSunterScorer,
    import_splink_scorer,
)
from vectorer.comparisons import make_comparison


def test_identical_pair_scores_high_and_distinct_pair_scores_low(fs_scorer):
    left = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "john@x.com"}
    same = {**left}
    different = {"first_name": "zoe", "last_name": "khan", "date_of_birth": "1999-01-01", "email": "zoe@y.com"}

    assert fs_scorer.score(left, same) > 0.99
    assert fs_scorer.score(left, different) < 0.5


def test_scorer_matches_defaults_on_email_exact():
    """An exact email match under default m/u reproduces the 0.0929 posterior.

    ``idempotent=False`` is passed so the raw calibrated m/u math is tested
    (with the reflexivity fix the same-content pair would otherwise be forced
    to 1.0).
    """
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [make_comparison("email_comparison", col_name="email")],
        prior=1e-4,
        idempotent=False,
    )
    left = {"email": "john.smith@example.com"}
    right = {"email": "john.smith@example.com"}
    # Exact-match level has weight 10, so posterior = sigmoid(log(1e-4) + log(2^10)) = 0.0929.
    assert scorer.score(left, right) == pytest.approx(0.0929, abs=1e-4)


def test_identical_pairs_are_reflexive_thin_records():
    """Idempotence (r ~ r): content-identical pairs score 1.0, even thin ones.

    Without the reflexivity fix a record whose comparison fields are mostly
    missing would score against itself below the threshold (its self-posterior
    is the prior for all-None fields).
    """
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [
            make_comparison("jaro_winkler_at_thresholds", col_name="first_name"),
            make_comparison("jaro_winkler_at_thresholds", col_name="last_name"),
            make_comparison("date_of_birth_comparison", col_name="date_of_birth"),
            make_comparison("email_comparison", col_name="email"),
        ],
        threshold=0.85,
    )
    all_none = {"first_name": None, "last_name": None, "date_of_birth": None, "email": None}
    thin = {"first_name": "john", "last_name": None, "date_of_birth": None, "email": None}
    # Reflexive regardless of how thin the record is.
    assert scorer.score(all_none, dict(all_none)) == 1.0
    assert scorer.score(thin, dict(thin)) == 1.0
    # batch forms too
    assert scorer.score_batch(thin, [dict(thin)])[0] == 1.0
    assert scorer.score_pairs([thin], [dict(thin)])[0] == 1.0
    # score_and_weight_batch returns the same posterior
    post, weight = scorer.score_and_weight_batch(thin, [dict(thin)])
    assert post[0] == 1.0
    assert np.isfinite(weight[0])


def test_identical_pairs_are_reflexive_after_disable_flag():
    """idempotent=False restores the raw calibrated posterior for same-content pairs."""
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [make_comparison("email_comparison", col_name="email")],
        prior=1e-4,
        idempotent=False,
    )
    left = {"email": "a@b.com"}
    assert scorer.score(left, dict(left)) == pytest.approx(0.0929, abs=1e-4)


def test_identical_pair_mask_ignores_non_compared_fields():
    """Only the compared columns decide content-equality, not extra attributes."""
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [make_comparison("email_comparison", col_name="email")],
        threshold=0.85,
    )
    a = {"email": "a@b.com", "note": "original"}
    b = {"email": "a@b.com", "note": "edited copy"}  # same compared field
    assert scorer.score(a, b) == 1.0


def test_idempotence_persists_in_round_trip(tmp_path):
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [make_comparison("email_comparison", col_name="email")], idempotent=True
    )
    path = tmp_path / "scorer.json"
    scorer.save(path)
    loaded = FellegiSunterScorer.load(path)
    assert loaded.idempotent is True
    thin = {"email": None}
    assert loaded.score(thin, dict(thin)) == 1.0


def test_score_batch_aligned_with_candidates(fs_scorer):
    left = {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None}
    candidates = [
        {"first_name": "mary", "last_name": "jones", "date_of_birth": "1990-11-03", "email": None},
        {"first_name": "bob", "last_name": "white", "date_of_birth": "1950-05-05", "email": "b@x.com"},
    ]
    posteriors = fs_scorer.score_batch(left, candidates)
    assert len(posteriors) == 2
    assert posteriors[0] > posteriors[1]


def test_match_weight_is_finite_for_certain_matches(fs_scorer):
    left = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "a@b.com"}
    weights = fs_scorer.match_weight_batch(left, [dict(left)])
    assert np.isfinite(weights[0])
    assert weights[0] > 10.0


def test_score_pairs_vectorised(fs_scorer):
    left = {"first_name": "a", "last_name": "b", "date_of_birth": "2000-01-01", "email": None}
    rights = [
        {"first_name": "a", "last_name": "b", "date_of_birth": "2000-01-01", "email": None},
        {"first_name": "zz", "last_name": "zz", "date_of_birth": "2001-02-02", "email": "x@y.com"},
    ]
    probs = fs_scorer.score_pairs([left, left], rights)
    weights = fs_scorer.match_weight_pairs([left, left], rights)
    assert probs.shape == (2,)
    assert probs[0] > probs[1]
    assert np.isfinite(weights).all()


def test_from_settings_dicts_round_trip(fs_scorer):
    data = fs_scorer.to_settings()
    restored = FellegiSunterScorer.from_settings(data)
    left = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com"}
    assert abs(restored.score(left, dict(left)) - fs_scorer.score(left, dict(left))) < 1e-6


def test_save_load(tmp_path, fs_scorer):
    path = tmp_path / "scorer.json"
    fs_scorer.save(path)
    loaded = FellegiSunterScorer.load(path)
    left = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "j@x.com"}
    assert abs(loaded.score(left, dict(left)) - fs_scorer.score(left, dict(left))) < 1e-6


def test_calibrate_from_pairs(fs_scorer):
    pairs = []
    for i in range(20):
        first, last, dob = f"name{i}", f"surname{i}", f"1980-01-{i % 27 + 1:02d}"
        pairs.append({
            "is_match": 1,
            "first_name_l": first, "first_name_r": first,
            "last_name_l": last, "last_name_r": last,
            "date_of_birth_l": dob, "date_of_birth_r": dob,
            "email_l": None, "email_r": None,
        })
    for i in range(40):
        pairs.append({
            "is_match": 0,
            "first_name_l": f"left{i}", "first_name_r": f"right{i}",
            "last_name_l": f"l{i}", "last_name_r": f"r{i}",
            "date_of_birth_l": "1990-01-01", "date_of_birth_r": "1991-01-01",
            "email_l": f"a{i}@x.com", "email_r": f"b{i}@x.com",
        })
    calibrated = fs_scorer.calibrate_from_pairs(pairs, smoothing=0.1)
    id_pair = {
        "first_name_l": "antonio", "first_name_r": "antonio",
        "last_name_l": "silva", "last_name_r": "silva",
        "date_of_birth_l": "1975-03-03", "date_of_birth_r": "1975-03-03",
        "email_l": None, "email_r": None,
    }
    posteriors = calibrated.score_pairs(
        [{"first_name": "antonio", "last_name": "silva", "date_of_birth": "1975-03-03", "email": None}],
        [{"first_name": "antonio", "last_name": "silva", "date_of_birth": "1975-03-03", "email": None}],
    )
    assert posteriors[0] > 0.5
    assert calibrated.to_dict()["comparisons"]
    assert calibrated.to_dict()["probability_two_random_records_match"] == pytest.approx(DEFAULT_PRIOR, abs=1e-12)


def test_fit_em_trains_mu(fs_scorer):
    records = []
    for i in range(12):
        records.append(
            {"first_name": f"name{i}", "last_name": f"surname{i}", "date_of_birth": f"19{i:02d}-01-01", "email": None}
        )
        records.append(
            {"first_name": f"name{i}", "last_name": f"surname{i}", "date_of_birth": f"19{i:02d}-01-01", "email": None}
        )
    trained = fs_scorer.fit_em(records, training_block_on=[("first_name",)], seed=7)
    left = {"first_name": "name0", "last_name": "surname0", "date_of_birth": "1900-01-01", "email": None}
    assert trained.score(left, dict(left)) > 0.9


def test_fit_em_prior_estimated(fs_scorer):
    records = []
    for i in range(8):
        records.append({"first_name": f"n{i}", "last_name": f"s{i}", "date_of_birth": f"19{i:02d}-01-01", "email": None})
        records.append({"first_name": f"n{i}", "last_name": f"s{i}", "date_of_birth": f"19{i:02d}-01-01", "email": None})
    trained = fs_scorer.fit_em(records, training_block_on=[("first_name",)], recall=1.0, seed=3)
    trained_settings = trained.to_settings()
    assert 0.0 < trained_settings["probability_two_random_records_match"] < 0.5
    # An exact duplicate should be a confident match under the trained model.
    left = {"first_name": "n0", "last_name": "s0", "date_of_birth": "1900-01-01", "email": None}
    assert trained.score(left, dict(left)) > 0.7
def test_import_splink_scorer_maps_mu_and_prior():
    from vectorer.comparisons import make_comparison

    splink_settings = {
        "comparisons": [
            {
                "output_column_name": "first_name",
                "comparison_levels": [
                    {"sql_condition": "x", "is_null_level": True},
                    {"sql_condition": "eq", "m_probability": 0.89, "u_probability": 0.0004},
                    {"sql_condition": "jw>0.9", "m_probability": 0.07, "u_probability": 0.002},
                    {"sql_condition": "jw>0.7", "m_probability": 0.03, "u_probability": 0.01},
                    {"sql_condition": "ELSE", "m_probability": 0.01, "u_probability": 0.4},
                ],
            },
        ],
        "probability_two_random_records_match": 1e-5,
    }
    native = [
        make_comparison("jaro_winkler_at_thresholds", col_name="first_name",
                        score_threshold_or_thresholds=[0.9, 0.7]),
    ]
    scorer = import_splink_scorer(splink_settings, native)
    assert scorer.prior == pytest.approx(1e-5)
    spec = scorer.comparisons[0].spec()
    vals = [(lv.m, lv.u) for lv in spec.levels if not lv.is_null]
    assert vals[0] == (0.89, 0.0004)
    assert vals[1] == (0.07, 0.002)
    assert vals[-1][1] == 0.4


def test_import_splink_scorer_tf_fields_carry_over():
    from vectorer.comparisons import make_comparison

    splink_settings = {
        "comparisons": [
            {
                "output_column_name": "email",
                "comparison_levels": [
                    {"sql_condition": "x", "is_null_level": True},
                    {"sql_condition": "eq", "m_probability": 0.9, "u_probability": 1e-5,
                     "tf_adjustment_weight": 1.0, "tf_minimum_u_value": 0.0,
                     "tf_adjustment_column": "email"},
                    {"sql_condition": "u_eq", "m_probability": 0.05, "u_probability": 0.0005},
                    {"sql_condition": "jw", "m_probability": 0.03, "u_probability": 0.002},
                    {"sql_condition": "ujw", "m_probability": 0.01, "u_probability": 0.005},
                    {"sql_condition": "ELSE", "m_probability": 0.01, "u_probability": 0.4},
                ],
            },
        ],
        "probability_two_random_records_match": 1e-5,
    }
    native = [make_comparison("email_comparison", col_name="email")]
    scorer = import_splink_scorer(splink_settings, native, base_records=[{"email": "a"}])
    # TF metadata lands on the exact level (no null).
    spec = scorer.comparisons[0].spec()
    exact = spec.levels[1]
    assert exact.tf_column == "email"
    assert exact.tf_weight == 1.0
    assert scorer.score({"email": "a"}, {"email": "a"}) == 1.0


def test_import_splink_scorer_rejects_missing_or_mismatched():
    from vectorer.comparisons import make_comparison

    native = [make_comparison("email_comparison", col_name="email")]
    # missing comparison
    with pytest.raises(ValueError, match="no Splink-trained comparison"):
        import_splink_scorer({"comparisons": [], "probability_two_random_records_match": 1e-5}, native)
    # level-count mismatch
    splink = {
        "comparisons": [{"output_column_name": "email", "comparison_levels": [
            {"sql_condition": "x", "is_null_level": True},
            {"sql_condition": "ELSE", "m_probability": 0.9, "u_probability": 0.1},
        ]}],
        "probability_two_random_records_match": 1e-5,
    }
    with pytest.raises(ValueError, match="levels"):
        import_splink_scorer(splink, native)


def test_fit_em_fixed_prior_freezes_base_rate():
    from vectorer.comparisons import make_comparison

    records = []
    for i in range(8):
        records.append({'first_name': f'n{i}', 'last_name': 's',
                        'date_of_birth': f'19{i:02d}-01-01', 'email': None, 'address': None})
        records.append({'first_name': f'n{i}', 'last_name': 's',
                        'date_of_birth': f'19{i:02d}-01-01', 'email': None, 'address': None})
    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]

    learned = FellegiSunterScorer.from_comparisons(comps).fit_em(
        records, training_block_on=[('first_name',)], max_iterations=5, seed=7)
    fixed = FellegiSunterScorer.from_comparisons(comps).fit_em(
        records, training_block_on=[('first_name',)], max_iterations=5, seed=7,
        fixed_prior=0.01)

    assert learned.to_settings()['probability_two_random_records_match'] != 0.01
    assert fixed.to_settings()['probability_two_random_records_match'] == pytest.approx(0.01)
    x = {'first_name': 'n0', 'last_name': 's', 'date_of_birth': '1900-01-01',
         'email': None, 'address': None}
    assert fixed.score(x, dict(x)) == 1.0  # idempotent ident match still holds


def test_fit_em_fixed_prior_and_prior_disagree_raises():
    from vectorer.comparisons import make_comparison

    records = [{'first_name': 'a', 'last_name': 'b', 'date_of_birth': '2000-01-01',
                'email': None, 'address': None}] * 4
    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    with pytest.raises(ValueError, match='disagree'):
        FellegiSunterScorer.from_comparisons(comps).fit_em(
            records, training_block_on=[('first_name',)], fixed_prior=0.01, prior=0.5, seed=1)


def test_recalibrate_prior_drops_enriched_prior():
    from vectorer.comparisons import make_comparison

    # Enriched training set: ~50% matches (every record duplicated once).
    enriched = []
    for i in range(20):
        base = {'first_name': f'n{i}', 'last_name': 's', 'date_of_birth': f'19{i % 50:02d}-01-01',
                'email': None, 'address': None}
        enriched.append(dict(base))
        enriched.append(dict(base))
    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    scorer = FellegiSunterScorer.from_comparisons(comps)
    enriched_model = scorer.fit_em(enriched, training_block_on=[('first_name',)],
                                   max_iterations=5, seed=7)
    prior_enriched = enriched_model.to_settings()['probability_two_random_records_match']

    # Recalibrate on a large, mostly-unrelated population (match share near 0).
    full = [{'first_name': f'z{i}', 'last_name': f'w{i}', 'date_of_birth': f'19{i % 50:02d}-01-01',
             'email': None, 'address': None} for i in range(200)]
    cal = enriched_model.recalibrate_prior(full, sample_size=10000, seed=1)
    prior_cal = cal.to_settings()['probability_two_random_records_match']
    assert prior_cal < prior_enriched
    # A truly unrelated population => prior collapses toward 0 (still > 0).
    assert prior_cal > 0


def test_recalibrate_prior_yancey_uses_count_ratio():
    from vectorer.comparisons import make_comparison

    enriched = []
    for i in range(20):
        base = {'first_name': f'n{i}', 'last_name': 's', 'date_of_birth': f'19{i % 50:02d}-01-01',
                'email': None, 'address': None}
        enriched.append(dict(base))
        enriched.append(dict(base))
    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    model = FellegiSunterScorer.from_comparisons(comps).fit_em(
        enriched, training_block_on=[('first_name',)], max_iterations=5, seed=7)
    # fit_em records the enriched EM metadata needed for Yancey's correction.
    assert model._em is not None and model._em["n_pairs"] > 0

    full = [{'first_name': f'z{i}', 'last_name': f'w{i}', 'date_of_birth': f'19{i % 50:02d}-01-01',
             'email': None, 'address': None} for i in range(200)]
    cal = model.recalibrate_prior(full, method="yancey")
    pi, n0 = model._em["pi"], model._em["n_pairs"]
    n = len(full)
    expected = (pi * n0) / (n * (n - 1) / 2)
    assert cal.to_settings()['probability_two_random_records_match'] == pytest.approx(expected)
    # Deterministic: no sampling, so seed must not matter.
    cal2 = model.recalibrate_prior(full, method="yancey", seed=999999)
    assert (cal2.to_settings()['probability_two_random_records_match']
            == cal.to_settings()['probability_two_random_records_match'])


def test_recalibrate_prior_empirical_resamples():
    from vectorer.comparisons import make_comparison

    enriched = []
    for i in range(20):
        base = {'first_name': f'n{i}', 'last_name': 's', 'date_of_birth': f'19{i % 50:02d}-01-01',
                'email': None, 'address': None}
        enriched.append(dict(base))
        enriched.append(dict(base))
    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    model = FellegiSunterScorer.from_comparisons(comps).fit_em(
        enriched, training_block_on=[('first_name',)], max_iterations=5, seed=7)
    full = [{'first_name': f'z{i}', 'last_name': f'w{i}', 'date_of_birth': f'19{i % 50:02d}-01-01',
             'email': None, 'address': None} for i in range(200)]
    y = model.recalibrate_prior(full, method="yancey")
    e = model.recalibrate_prior(full, method="empirical", sample_size=5000, seed=1)
    # Both drop the enriched prior, but the empirical resample depends on
    # scoring draws (its own value), while yancey is the deterministic ratio.
    assert e.to_settings()['probability_two_random_records_match'] < model.prior
    assert (y.to_settings()['probability_two_random_records_match']
            != e.to_settings()['probability_two_random_records_match'])


def test_recalibrate_prior_yancey_requires_em_metadata():
    from vectorer.comparisons import make_comparison

    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    # A plain scorer never ran fit_em => no EM metadata.
    plain = FellegiSunterScorer.from_comparisons(comps)
    full = [{'first_name': f'z{i}', 'last_name': 'w', 'date_of_birth': '2000-01-01',
             'email': None, 'address': None} for i in range(10)]
    with pytest.raises(ValueError, match="fit_em"):
        plain.recalibrate_prior(full, method="yancey")
    # Unknown method is rejected too.
    with pytest.raises(ValueError, match="unknown recalibrate_prior method"):
        plain.recalibrate_prior(full, method="bogus")


def test_fit_em_uses_fuzzy_blocking_for_perturbed_twins():
    """Case-flipped/typ'd twins must reach the blocked training pool.

    Regression: exact-string blocking on first_name never joins a perturbed
    twin (case flip/typo) to its base, so EM's candidate pool contained
    ~0 true matches and the learned m/u collapsed.
    """
    from vectorer.comparisons import make_comparison

    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    scorer = FellegiSunterScorer.from_comparisons(comps)
    # A base record + a case-flipped twin (the dominant perturbation).
    records = []
    for i in range(40):
        records.append({'first_name': f'Jessica{i}', 'last_name': f'Smith{i}',
                        'date_of_birth': f'19{i % 50:02d}-01-01', 'email': None,
                        'address': None})
        records.append({'first_name': f'jessicA{i}', 'last_name': f'Smith{i}',
                        'date_of_birth': f'19{i % 50:02d}-01-01', 'email': None,
                        'address': None})
    blocked = scorer._blocked_pairs(records, [('first_name',)], 10000,
                                    np.random.default_rng(0))
    # The exact-identical base names collide AND the case-flipped twins collide
    # (they map into the same fuzzy trigram bucket to the same base).
    assert len(blocked) > 0
    # At least one blocked pair joins index 0 (a base) with index 1 (its case-
    # flipped twin) or vice versa.
    joined_twin = any(
        {a, b} == {2 * i, 2 * i + 1} for a, b in blocked for i in range(40)
    )
    assert joined_twin


def test_fit_em_pi_is_bounded_below_one():
    """The blocked-pair match share must not run away to 1.0 even when the
    candidate pool is dominated by same-name non-matches."""
    from vectorer.comparisons import make_comparison

    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name')]
    scorer = FellegiSunterScorer.from_comparisons(comps)
    records = []
    for i in range(100):
        base = {'first_name': f'Name{i}', 'last_name': 'Same',
                'date_of_birth': f'19{i % 50:02d}-01-01',
                'email': f'u{i}@example.com', 'address': f'{i} Main St'}
        records.append(dict(base))
        records.append(dict(base))
    trained = scorer.fit_em(records, training_block_on=[('first_name',)],
                            max_iterations=10, seed=3)
    # pi is clipped away from exactly 1.0.
    assert trained._em["pi"] < 1.0
    # The uniform-posterior prior is finite and in (0, 1).
    assert 0.0 < trained.to_settings()[
        'probability_two_random_records_match'] < 1.0
    # Twin posterior is high (a duplicate scores near 1).
    q = {'first_name': 'Name7', 'last_name': 'Same', 'date_of_birth': '1907-01-01',
         'email': 'u7@example.com', 'address': '7 Main St'}
    assert trained.score(q, dict(q)) > 0.9


def test_fit_em_pool_contains_true_matches_with_dob_rule():
    """With a date_of_birth blocking rule, twins (which share DOB) reach the
    pool even when the name rule saturates the budget."""
    from vectorer.comparisons import make_comparison

    comps = [make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
             make_comparison('date_of_birth_comparison', col_name='date_of_birth')]
    scorer = FellegiSunterScorer.from_comparisons(comps)
    # 50 bases + 50 exact twins; the name rule yields many common cliques, but
    # the DOB rule (twins share exact DOB) must still yield matches.
    records = []
    for i in range(50):
        base = {'first_name': 'Common', 'last_name': f'L{i}',
                'date_of_birth': f'19{i % 50:02d}-01-01',
                'email': f'u{i}@ex.com', 'address': f'{i} St'}
        records.append(dict(base))
        records.append(dict(base))
    blocked = scorer._blocked_pairs(records, [('first_name',), ('date_of_birth',)],
                                    100000, np.random.default_rng(0))
    # The DOB-exact rule ties each twin to its own base.
    dob_twin_blocks = any(
        {a, b} == {2 * i, 2 * i + 1} for a, b in blocked for i in range(50)
    )
    assert dob_twin_blocks


def test_blocked_pairs_multi_column_rule_exact_conjunction():
    """A multi-column rule joins only records equal on every key column."""
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
    ])
    records = [
        {'first_name': 'a', 'last_name': 'smith', 'date_of_birth': '2000-01-01'},
        {'first_name': 'a', 'last_name': 'jones', 'date_of_birth': '2000-01-01'},
        {'first_name': 'a', 'last_name': 'smith', 'date_of_birth': '2000-01-01'},
        {'first_name': 'b', 'last_name': 'smith', 'date_of_birth': '2000-01-01'},
    ]
    blocked = scorer._blocked_pairs(
        records, [('first_name', 'last_name')], 10000, np.random.default_rng(0))
    # Only records 0 and 2 share BOTH first_name and last_name.
    assert (0, 2) in {tuple(sorted(p)) for p in blocked}
    assert not any(b == 3 for _, b in blocked)


def test_sample_all_pairs_full_enumeration_and_too_small():
    """_sample_all_pairs enumerates all pairs when the cap exceeds the total,
    returns [] for tiny n, and never double-returns an unordered pair."""
    from vectorer.scoring._candidates import _sample_all_pairs

    rng = np.random.default_rng(42)
    assert _sample_all_pairs(0, 10, rng) == []
    assert _sample_all_pairs(1, 10, rng) == []
    full = _sample_all_pairs(4, 100, rng)
    assert len(full) == 6  # C(4,2)
    assert set(full) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    capped = _sample_all_pairs(1000, 10, rng)
    assert len(capped) == 10
    assert len(set(capped)) == 10
    assert all(a < b for a, b in capped)


def test_uniform_prior_estimate_returns_nan_on_empty_population():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('email_comparison', col_name='email'),
    ])
    assert scorer._uniform_prior_estimate([], 100, np.random.default_rng(0)) != scorer._uniform_prior_estimate([], 100, np.random.default_rng(0))


def test_recalibrate_empirical_returns_self_on_empty_population():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('email_comparison', col_name='email'),
    ])
    out = scorer._recalibrate_empirical([], sample_size=100, seed=1)
    assert out is scorer


def test_values_equal_handles_arrays_lists_and_scalars():
    from vectorer.scoring._math import _values_equal

    import numpy as _np

    assert _values_equal(_np.array([1, 2]), _np.array([1, 2]))
    assert not _values_equal(_np.array([1, 2]), _np.array([1, 3]))
    assert not _values_equal(_np.array([1, 2]), [1, 2])  # array vs list mismatch
    assert _values_equal([1, 2], [1, 2])
    assert _values_equal([1, 2], (1, 2))  # list/tuple compare element-wise
    assert not _values_equal([1, 2], [1, 3])
    assert _values_equal("abc", "abc")
    assert not _values_equal("abc", "abd")
    assert not _values_equal(None, 0)


def test_level_log_bayes_factors_saturates_at_zero_mu():
    """m<=0 or u<=0 levels saturate to the log clip rather than log(0)."""
    from vectorer.scoring._levels import _level_log_bayes_factors

    spec = make_comparison('email_comparison', col_name='email').spec()
    for level, m, u in zip(spec.levels, [1e-8, 0.0, 0.9, 0.9, 0.9], [1e-8, 0.9, 0.0, 0.5, 0.1]):
        level.m = m
        level.u = u
    factors = _level_log_bayes_factors(spec)
    assert all(f != float('-inf') for f in factors)
    assert all(f != float('inf') for f in factors)


def test_combined_bayes_equals_exp_of_log_total():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
    ])
    left = {'first_name': 'john', 'last_name': 's', 'date_of_birth': '2000-01-01', 'email': None}
    pv = scorer._record_pair_values([left], [dict(left)])
    assert scorer._combined_bayes(pv) == pytest.approx(
        np.exp(scorer._log_total_bayes(pv)))
    assert np.isfinite(scorer._combined_bayes(pv)).all()


def test_empty_candidates_public_paths():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('email_comparison', col_name='email'),
    ])
    left = {'email': 'a@x.com'}
    assert scorer.score_batch(left, []) == pytest.approx(np.asarray([]))
    assert scorer.match_weight_batch(left, []) == pytest.approx(np.asarray([]))
    pos, w = scorer.score_and_weight_batch(left, [])
    assert len(pos) == 0 and len(w) == 0
    assert scorer.score_pairs([], []) == pytest.approx(np.asarray([]))
    assert scorer.match_weight_pairs([], []) == pytest.approx(np.asarray([]))


def test_fit_em_requires_comparisons_and_candidate_pairs():
    from vectorer.scoring import FellegiSunterScorer

    empty = FellegiSunterScorer.from_comparisons([])
    with pytest.raises(ValueError, match="no comparisons"):
        empty.fit_em([{'first_name': 'a'}])

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
    ])
    # All block values are None -> no candidate pairs.
    with pytest.raises(RuntimeError, match="no blocking-rule candidate pairs"):
        scorer.fit_em([{'first_name': None}, {'first_name': None}])


def test_fit_em_prior_override_reports_value():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
    ])
    records = []
    for i in range(40):
        base = {'first_name': f'Name{i}', 'last_name': 's',
                'date_of_birth': f'19{i % 40:02d}-01-01', 'email': None}
        records.append(dict(base))
        records.append(dict(base))
    trained = scorer.fit_em(records, training_block_on=[('first_name',)],
                            max_iterations=5, seed=1, prior=0.1234)
    assert trained.to_settings()['probability_two_random_records_match'] == pytest.approx(0.1234)


def test_union_expand_no_set_fields_returns_single_pair():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
    ])
    left = {'first_name': 'john', 'last_name': 's', 'date_of_birth': '2000-01-01', 'email': None}
    rows = scorer._union_expand(left, dict(left), ['first_name'])
    assert rows == [(left, dict(left))]

    # A union pair whose other field has no set values still expands only the
    # set field; a record with no frozensets among its compared fields is
    # returned unchanged.
    right = {'first_name': frozenset({'john', 'jane'}), 'last_name': 's',
             'date_of_birth': '2000-01-01', 'email': None}
    expanded = scorer._union_expand(left, right, ['first_name', 'last_name'])
    assert len(expanded) == 2
    assert {r['first_name'] for _, r in expanded} == {'john', 'jane'}


def test_score_pairs_with_union_lift_returns_max():
    from vectorer.scoring import FellegiSunterScorer

    scorer = FellegiSunterScorer.from_comparisons([
        make_comparison('jaro_winkler_at_thresholds', col_name='first_name'),
        make_comparison('date_of_birth_comparison', col_name='date_of_birth'),
    ])
    # A record holding a frozenset of alternatives should score as the max over
    # its members, and the non-union sibling method must not be called.
    member = {'first_name': 'john', 'last_name': 's', 'date_of_birth': '2000-01-01', 'email': None}
    union = {'first_name': frozenset({'john', 'zzzz'}), 'last_name': 's',
             'date_of_birth': '2000-01-01', 'email': None}
    assert scorer.score(union, member) == pytest.approx(1.0)


def test_from_comparisons_accepts_resolved_dicts_and_specs():
    from vectorer.scoring import FellegiSunterScorer

    comp = make_comparison('email_comparison', col_name='email')
    # _as_specs handles Comparison, ComparisonSpec, and dict forms.
    from vectorer.comparisons import ComparisonSpec

    spec = comp.spec()
    as_dict = comp.resolved()
    for source in ([comp], [spec], [as_dict]):
        scorer = FellegiSunterScorer.from_comparisons(source)
        assert scorer.comparisons
    with pytest.raises(TypeError, match="expected Comparison"):
        FellegiSunterScorer.from_comparisons([object()])


def test_weights_tf_table_returns_none_for_empty_column():
    from vectorer.scoring._weights import _build_tf_table

    assert _build_tf_table('email', [{'email': None}, {'email': None}]) is None
    assert _build_tf_table('email', []) is None
    t = _build_tf_table('email', [{'email': 'a@x.com'}, {'email': 'b@x.com'}, {'email': 'a@x.com'}])
    assert t['a@x.com'] == pytest.approx(2 / 3)
