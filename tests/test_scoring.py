"""Tests for the Fellegi-Sunter scoring engine (native NumPy implementation)."""

import numpy as np
import pytest

from vectorer.scoring import (
    DEFAULT_PRIOR,
    FellegiSunterScorer,
)


def test_identical_pair_scores_high_and_distinct_pair_scores_low(fs_scorer):
    left = {"first_name": "john", "last_name": "smith", "date_of_birth": "1985-06-15", "email": "john@x.com"}
    same = {**left}
    different = {"first_name": "zoe", "last_name": "khan", "date_of_birth": "1999-01-01", "email": "zoe@y.com"}

    assert fs_scorer.score(left, same) > 0.99
    assert fs_scorer.score(left, different) < 0.5


def test_scorer_matches_splink_defaults_on_email_exact():
    """An exact email match under default m/u reproduces Splink's posterior."""
    from vectorer.comparisons import make_comparison

    scorer = FellegiSunterScorer.from_comparisons(
        [make_comparison("email_comparison", col_name="email")],
        prior=1e-4,
    )
    left = {"email": "john.smith@example.com"}
    right = {"email": "john.smith@example.com"}
    # Exact-match level has weight 10, so posterior = sigmoid(log(1e-4) + log(2^10)) = 0.0929.
    assert scorer.score(left, right) == pytest.approx(0.0929, abs=1e-4)


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