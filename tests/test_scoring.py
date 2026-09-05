"""Tests for the Fellegi-Sunter scoring engine (native NumPy implementation)."""

import numpy as np
import pytest

from vectorer.scoring import (
    DEFAULT_PRIOR,
    FellegiSunterScorer,
    import_splink_scorer,
)


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
