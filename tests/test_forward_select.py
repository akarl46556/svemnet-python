"""Deterministic forward selection tests."""

import math

import numpy as np
import pytest

from svemnet.forward import forward_select


def _toy(n=40, seed=7, sd=0.25):
    rng = np.random.default_rng(seed)
    X1, X2, X3 = rng.normal(size=(3, n))
    y = 1 + 2 * X1 - 1.5 * X2 + 1.2 * X1 * X2 + rng.normal(0, sd, n)
    X = np.column_stack([X1, X2, X3, X1 * X2, X1 * X3, X2 * X3])
    names = ("X1", "X2", "X3", "X1:X2", "X1:X3", "X2:X3")
    return X, y, names


def _closed_form(criterion, n, rss, k):
    """k counts fitted coefficients including the intercept."""
    K = k + 1
    if criterion == "AICc":
        return n * math.log(rss / n) + 2 * K + 2 * K * (K + 1) / (n - K - 1)
    if criterion == "AIC":
        return n * math.log(rss / n) + 2 * K
    return n * math.log(rss / n) + K * math.log(n)


class TestForwardSelect:
    def test_recovers_true_terms_and_matches_lstsq(self):
        X, y, names = _toy()
        result = forward_select(X, y, feature_names=names)
        assert {"X1", "X2", "X1:X2"} <= set(result.selected_terms)
        sel = [0] + [1 + names.index(t) for t in result.selected_terms]
        Xint = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xint[:, sel], y, rcond=None)
        np.testing.assert_allclose(result.coefficients[sel], beta, rtol=1e-8)
        preds = result.predict(X)
        np.testing.assert_allclose(
            preds, Xint @ result.coefficients, rtol=1e-10
        )

    @pytest.mark.parametrize("criterion", ["AICc", "AIC", "BIC"])
    def test_criterion_value_matches_closed_form(self, criterion):
        X, y, names = _toy()
        result = forward_select(X, y, criterion=criterion, feature_names=names)
        k = 1 + sum(len((j,)) for j, c in enumerate(result.coefficients[1:]) if c != 0)
        expected = _closed_form(
            criterion, result.n_obs, result.residual_sum_of_squares, k
        )
        assert np.isclose(result.criterion_value, expected, rtol=1e-10)

    def test_path_improves_monotonically(self):
        X, y, names = _toy()
        result = forward_select(X, y, feature_names=names)
        values = [step["criterion_value"] for step in result.selection_path]
        assert len(values) >= 2
        assert all(b < a for a, b in zip(values, values[1:]))
        assert all(step["improvement"] > 0 for step in result.selection_path)

    def test_deterministic(self):
        X, y, names = _toy()
        r1 = forward_select(X, y, feature_names=names)
        r2 = forward_select(X, y, feature_names=names)
        np.testing.assert_array_equal(r1.coefficients, r2.coefficients)
        assert r1.selected_terms == r2.selected_terms

    def test_bic_selects_no_more_terms_than_aic(self):
        X, y, names = _toy()
        aic = forward_select(X, y, criterion="AIC", feature_names=names)
        bic = forward_select(X, y, criterion="BIC", feature_names=names)
        assert len(bic.selected_terms) <= len(aic.selected_terms)

    def test_groups_enter_jointly(self):
        rng = np.random.default_rng(3)
        n = 60
        Fb = (rng.random(n) < 0.5).astype(float)
        Fc = ((rng.random(n) < 0.5) & (Fb == 0)).astype(float)
        X1 = rng.normal(size=n)
        y = 1 + 2 * X1 + 2 * Fb - 1.5 * Fc + rng.normal(0, 0.3, n)
        X = np.column_stack([X1, Fb, Fc])
        result = forward_select(
            X, y,
            groups={"X1": [0], "F": [1, 2]},
            feature_names=("X1", "F[b]", "F[c]"),
        )
        assert "F" in result.selected_terms
        # both contrast columns nonzero because the block entered as a unit
        assert result.coefficients[2] != 0 and result.coefficients[3] != 0
        # model size jumped by 2 at the F step
        f_steps = [s for s in result.selection_path if s["term"] == "F"]
        assert len(f_steps) == 1

    def test_exactly_collinear_candidate_is_skipped(self):
        rng = np.random.default_rng(9)
        n = 40
        A = rng.uniform(0.1, 0.6, n)
        B = rng.uniform(0.1, 0.35, n)
        C = 1 - A - B  # collinear with intercept + A + B
        y = 2 + 3 * A - 2 * B + rng.normal(0, 0.2, n)
        X = np.column_stack([A, B, C])
        result = forward_select(X, y, feature_names=("A", "B", "C"))
        active = np.sum(result.coefficients[1:] != 0)
        assert active <= 2
        assert np.all(np.isfinite(result.coefficients))

    def test_near_perfect_candidates_compared_by_exact_rss(self):
        # Regression test from the R-port review: the projected RSS update is
        # cancellation noise for near-perfect fits; the exact-refit fallback
        # must pick the candidate with the smaller true RSS.
        rng = np.random.default_rng(1)
        n = 20
        cB = rng.normal(size=n)
        y = cB + 4e-9 * rng.normal(size=n)
        c2 = y + 1e-9 * rng.normal(size=n)
        X = np.column_stack([cB, c2])
        result = forward_select(X, y, feature_names=("cB", "c2"))
        assert result.selected_terms[0] == "c2"
        assert all(step["rss"] > 0 for step in result.selection_path)

    def test_badly_scaled_column_survives(self):
        # Timestamp-scale predictor: projected-norm ratio ~5e-8 vs intercept.
        n = 40
        rng = np.random.default_rng(8)
        ts = 1.7e9 + np.linspace(0, 300, n)
        x2 = rng.normal(size=n)
        y = 3 + 0.05 * (ts - ts.mean()) + 2 * x2 + rng.normal(0, 0.2, n)
        X = np.column_stack([ts, x2])
        result = forward_select(X, y, feature_names=("ts", "x2"))
        assert "ts" in result.selected_terms
        preds = result.predict(X)
        assert np.corrcoef(preds, y)[0, 1] ** 2 > 0.9

    def test_small_n_warns_and_returns_intercept_only(self):
        X = np.array([[0.1], [0.5], [0.9]])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="undefined"):
            result = forward_select(X, y)
        assert result.selected_terms == ()
        assert result.coefficients[1] == 0.0

    def test_constant_response_selects_nothing(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(20, 2))
        y = np.full(20, 5.0)
        result = forward_select(X, y)
        assert result.selected_terms == ()
        np.testing.assert_allclose(result.coefficients[0], 5.0)

    def test_input_validation(self):
        X, y, names = _toy()
        with pytest.raises(ValueError):
            forward_select(X, y, criterion="Cp")
        with pytest.raises(ValueError):
            forward_select(X, y, improvement_tolerance=-1.0)
        with pytest.raises(ValueError):
            forward_select(X, y, groups={"a": [0], "b": [0]})  # overlap
        with pytest.raises(ValueError):
            forward_select(X, y, groups={"a": [99]})  # out of range
