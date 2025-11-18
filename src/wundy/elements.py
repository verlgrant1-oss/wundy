from __future__ import annotations

from typing import Mapping, Any

import numpy as np
from numpy.typing import NDArray

from .materials import (
    linear_elastic_tangent,
    linear_elastic_stress,
    neo_hooke_tangent,
    neo_hooke_stress,
)
# ---------------------------------------------------------------------
# Material selection for stiffness and stress
# ---------------------------------------------------------------------
def material_tangent(material, strain):
    mtype = material["type"].upper()
    if mtype == "ELASTIC":
        return linear_elastic_tangent(material, strain)
    if mtype == "NEO_HOOKE":
        return neo_hooke_tangent(material, strain)
    raise NotImplementedError(f"Unknown material type {mtype}")


def material_stress(material, strain):
    mtype = material["type"].upper()
    if mtype == "ELASTIC":
        return linear_elastic_stress(material, strain)
    if mtype == "NEO_HOOKE":
        return neo_hooke_stress(material, strain)
    raise NotImplementedError(f"Unknown material type {mtype}")



def gauss_points_1d(ngauss: int) -> tuple[NDArray[float], NDArray[float]]:
    """
    Return Gauss–Legendre points and weights on [-1, 1] for 1D integration.
    Supported: 1 or 2 points.
    """
    if ngauss == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif ngauss == 2:
        g = 1.0 / np.sqrt(3.0)
        xi = np.array([-g, g])
        w = np.array([1.0, 1.0])
    else:
        raise NotImplementedError(
            f"gauss_points_1d supports ngauss = 1 or 2, got {ngauss}"
        )
    return xi, w


def t1d1_shape_functions(xi: float) -> tuple[NDArray[float], NDArray[float]]:
    """
    Shape functions and derivatives for a 2-node 1D bar element (T1D1).
    """
    N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])
    dN_dxi = np.array([-0.5, 0.5])
    return N, dN_dxi


def t1d1_element_stiffness(
    x_e: NDArray[float],
    area: float,
    material: Mapping[str, Any],
    ngauss: int = 2,
) -> NDArray[float]:
    """
    2x2 element stiffness for a 1D bar (T1D1) using Gauss quadrature.
    """
    x_e = np.asarray(x_e, dtype=float).reshape(-1)
    if x_e.size != 2:
        raise ValueError("t1d1_element_stiffness expects 2 nodes")

    A = float(area)
    if A <= 0.0:
        raise ValueError(f"Area must be positive, got {A}")

    x1, x2 = x_e
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero-length element in t1d1_element_stiffness")

    J = L / 2.0
    detJ = abs(J)

    # B is constant for linear 2-node bar
    _, dN_dxi = t1d1_shape_functions(0.0)
    dN_dx = dN_dxi / J
    B = dN_dx.reshape(1, -1)  # (1 x 2)

    Et = linear_elastic_tangent(material, strain=0.0)

    ke = np.zeros((2, 2), dtype=float)
    xi_g, w_g = gauss_points_1d(ngauss)
    for w in w_g:
        ke += B.T @ (Et * A * B) * detJ * w

    return ke


def t1d1_element_internal_force(
    x_e: NDArray[float],
    u_e: NDArray[float],
    area: float,
    material: Mapping[str, Any],
    ngauss: int = 2,
) -> NDArray[float]:
    """
    2x1 internal force vector for a 1D bar (T1D1) using Gauss quadrature.
    """
    x_e = np.asarray(x_e, dtype=float).reshape(-1)
    u_e = np.asarray(u_e, dtype=float).reshape(-1)

    if x_e.size != 2 or u_e.size != 2:
        raise ValueError("t1d1_element_internal_force expects 2-node element")

    A = float(area)
    if A <= 0.0:
        raise ValueError(f"Area must be positive, got {A}")

    x1, x2 = x_e
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero-length element in t1d1_element_internal_force")

    J = L / 2.0
    detJ = abs(J)

    f_int = np.zeros(2, dtype=float)
    xi_g, w_g = gauss_points_1d(ngauss)

    for xi, w in zip(xi_g, w_g):
        _, dN_dxi = t1d1_shape_functions(xi)
        dN_dx = dN_dxi / J
        B = dN_dx.reshape(1, -1)

        strain = float(B @ u_e)
        sigma = linear_elastic_stress(material, strain)

        f_int += (B.T * (sigma * A) * detJ * w).reshape(2)

    return f_int


def t1d1_element_uniform_load(
    x_e: NDArray[float],
    area: float,
    q: float,
    direction: float,
    ngauss: int = 2,
) -> NDArray[float]:
    """
    2x1 external force vector for a uniform line/body load on a T1D1 element.
    """
    x_e = np.asarray(x_e, dtype=float).reshape(-1)
    if x_e.size != 2:
        raise ValueError("t1d1_element_uniform_load expects 2-node element")

    A = float(area)
    x1, x2 = x_e
    L = x2 - x1
    if np.isclose(L, 0.0):
        raise ValueError("Zero-length element in t1d1_element_uniform_load")

    J = L / 2.0
    detJ = abs(J)

    dir_sign = float(np.sign(direction))
    if dir_sign == 0.0:
        raise ValueError(f"direction must be ±1, got {direction}")

    q_eff = float(q) * dir_sign * A

    f_ext = np.zeros(2, dtype=float)
    xi_g, w_g = gauss_points_1d(ngauss)

    for xi, w in zip(xi_g, w_g):
        N, _ = t1d1_shape_functions(xi)
        f_ext += N * q_eff * detJ * w

    return f_ext

# ---------------------------------------------------------------------
# Element Residual (for Newton solver)
# ---------------------------------------------------------------------
def t1d1_element_residual(
    x_e: NDArray[float],
    u_e: NDArray[float],
    area: float,
    material: Mapping[str, Any],
    f_ext_e: NDArray[float],
    ngauss: int = 2,
) -> NDArray[float]:
    """
    Compute the element residual vector:
        r_e = f_int - f_ext
    where:
        f_int = internal force (from constitutive model)
        f_ext = external nodal loads acting on the element
    """
    f_int = t1d1_element_internal_force(x_e, u_e, area, material, ngauss)
    r_e = f_int - f_ext_e
    return r_e
