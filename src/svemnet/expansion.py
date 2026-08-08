"""Formula-string builders for standard DOE response-surface expansions.

The counterpart of the R SVEMnet helper ``bigexp_terms()``: build the
right-hand side of a response-surface formula once and reuse it across
responses. These helpers only build strings — pair them with
:func:`svemnet.formula.svem` / :func:`svemnet.formula.forward_aicc` (which
require the ``svemnet[formula]`` extra) or any formulaic workflow.
"""

from __future__ import annotations

from typing import Sequence


def response_surface_formula(
    response: str,
    continuous: Sequence[str],
    nominal: Sequence[str] = (),
    *,
    interaction_order: int = 2,
    polynomial_order: int = 2,
) -> str:
    """Build a response-surface model formula.

    Includes all factors' main effects, interactions among all factors up to
    ``interaction_order``, and pure polynomial powers of the continuous
    factors up to ``polynomial_order``. (Unlike R SVEMnet's
    ``bigexp_terms()``, no partial-cubic cross terms are generated.)

    >>> response_surface_formula("y", ["X1", "X2"], ["F"])
    'y ~ (X1 + X2 + F)**2 + I(X1**2) + I(X2**2)'
    """
    continuous = list(continuous)
    nominal = list(nominal)
    if not continuous and not nominal:
        raise ValueError("at least one factor is required")
    seen: set[str] = set()
    for name in continuous + nominal:
        if not isinstance(name, str) or not name:
            raise ValueError("factor names must be non-empty strings")
        if name in seen:
            raise ValueError(f"duplicate factor name {name!r}")
        seen.add(name)
    if response in seen:
        raise ValueError("the response cannot also be a factor")
    if interaction_order < 1:
        raise ValueError("interaction_order must be at least 1")
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be at least 1")

    all_factors = " + ".join(continuous + nominal)
    if interaction_order == 1 or len(continuous) + len(nominal) == 1:
        rhs = all_factors
    else:
        rhs = f"({all_factors})**{interaction_order}"
    powers = [
        f"I({name}**{degree})"
        for degree in range(2, polynomial_order + 1)
        for name in continuous
    ]
    if powers:
        rhs = rhs + " + " + " + ".join(powers)
    return f"{response} ~ {rhs}"


__all__ = ["response_surface_formula"]
