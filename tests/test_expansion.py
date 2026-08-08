"""Formula-string builder tests."""

import pytest

from svemnet import response_surface_formula


class TestResponseSurfaceFormula:
    def test_quadratic_with_nominal(self):
        assert (
            response_surface_formula("y", ["X1", "X2"], ["F"])
            == "y ~ (X1 + X2 + F)**2 + I(X1**2) + I(X2**2)"
        )

    def test_cubic_powers(self):
        formula = response_surface_formula(
            "y", ["A", "B"], polynomial_order=3
        )
        assert "I(A**2)" in formula and "I(A**3)" in formula
        assert "I(B**2)" in formula and "I(B**3)" in formula

    def test_main_effects_only(self):
        assert (
            response_surface_formula("y", ["A"], interaction_order=1,
                                     polynomial_order=1)
            == "y ~ A"
        )

    def test_single_factor_skips_interaction_wrapper(self):
        assert response_surface_formula("y", ["A"]) == "y ~ A + I(A**2)"

    def test_nominal_gets_no_powers(self):
        formula = response_surface_formula("y", ["A"], ["F"])
        assert "I(F" not in formula

    def test_non_identifier_names_are_backticked(self):
        formula = response_surface_formula("y", ["flow-rate", "temp C"])
        assert "`flow-rate`" in formula
        assert "`temp C`" in formula
        assert "I(`flow-rate`**2)" in formula
        assert response_surface_formula("out come", ["A"]).startswith(
            "`out come` ~"
        )
        with pytest.raises(ValueError, match="backtick"):
            response_surface_formula("y", ["bad`name"])

    def test_validation(self):
        with pytest.raises(ValueError):
            response_surface_formula("y", [])
        with pytest.raises(ValueError):
            response_surface_formula("y", ["A", "A"])
        with pytest.raises(ValueError):
            response_surface_formula("y", ["y"])
        with pytest.raises(ValueError):
            response_surface_formula("y", ["A"], interaction_order=0)
        with pytest.raises(ValueError):
            response_surface_formula("y", ["A"], polynomial_order=0)
