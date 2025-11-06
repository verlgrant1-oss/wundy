from __future__ import annotations

from typing import Mapping, Any


def linear_elastic_tangent(material: Mapping[str, Any], strain: float) -> float:
    """
    Return the material tangent stiffness dσ/dε for a 1D linear-elastic model.

    Parameters
    ----------
    material
        Material dictionary as parsed from the input file. Must contain:
        - ``"type"``: currently expected to be ``"ELASTIC"``
        - ``"parameters"["E"]``: Young's modulus (E > 0)
    strain
        Current axial strain ε at the material point. For linear elasticity,
        the tangent stiffness is independent of strain but the argument is
        included to keep the interface compatible with non-linear models.

    Returns
    -------
    float
        Axial tangent stiffness dσ/dε (equal to E for a linear-elastic bar).
    """
    mtype = material.get("type", "").upper()
    if mtype != "ELASTIC":
        raise NotImplementedError(
            f"Material type {mtype!r} not implemented in linear_elastic_tangent"
        )

    E = material["parameters"]["E"]
    if E <= 0.0:
        raise ValueError(f"Young's modulus must be positive, got {E}")

    return float(E)


def linear_elastic_stress(material: Mapping[str, Any], strain: float) -> float:
    """
    Compute axial stress for a 1D linear-elastic material σ = E ε.

    Parameters
    ----------
    material
        Material dictionary as parsed from the input file. Must contain:
        - ``"type"``: currently expected to be ``"ELASTIC"``
        - ``"parameters"["E"]``: Young's modulus (E > 0)
    strain
        Axial strain ε at the material point.

    Returns
    -------
    float
        Axial Cauchy stress σ.
    """
    Et = linear_elastic_tangent(material, strain)
    return Et * float(strain)

