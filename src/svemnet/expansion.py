"""Formula-string builders for standard DOE response-surface expansions.

The counterpart of the R SVEMnet helper ``bigexp_terms()``: build the
right-hand side of a response-surface formula once and reuse it across
responses. These helpers only build strings — pair them with
:func:`svemnet.formula.svem` / :func:`svemnet.formula.forward_aicc` (which
require the ``svemnet[formula]`` extra) or any formulaic workflow.
"""

from __future__ import annotations

import keyword
from typing import Sequence


def _quote(name: str) -> str:
    """Backtick-quote factor names that are not plain Python identifiers.

    Formulaic requires backticks for names with spaces, hyphens, or other
    operator characters; without quoting, ``a-b`` would silently parse as
    the term ``a`` minus the term ``b``.
    """
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    if "`" in name:
        raise ValueError(f"factor name {name!r} may not contain backticks")
    return f"`{name}`"


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

    all_factors = " + ".join(_quote(f) for f in continuous + nominal)
    if interaction_order == 1 or len(continuous) + len(nominal) == 1:
        rhs = all_factors
    else:
        rhs = f"({all_factors})**{interaction_order}"
    powers = [
        f"I({_quote(name)}**{degree})"
        for degree in range(2, polynomial_order + 1)
        for name in continuous
    ]
    if powers:
        rhs = rhs + " + " + " + ".join(powers)
    return f"{_quote(response)} ~ {rhs}"


__all__ = ["response_surface_formula"]
