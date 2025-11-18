from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET, NEUMANN
from .elements import t1d1_element_stiffness, t1d1_element_uniform_load


def first_fe_code(
    coords: NDArray[float],
    blocks: list[dict],
    bcs: list[dict],
    dloads: list[dict],
    materials: dict[str, Any],
    block_elem_map: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    """
    Assemble and solve a 1D linear-elastic finite element bar problem.

    This version delegates element-level computations to modular procedures in
    :mod:`wundy.elements`:

    - ``t1d1_element_stiffness`` for element stiffness
    - ``t1d1_element_uniform_load`` for equivalent nodal forces from
      distributed loads

    The external interface and results are kept compatible with the original
    implementation, so existing tests and input files continue to work.
    """
    dof_per_node = 1
    num_node = coords.shape[0]
    num_dof = num_node * dof_per_node

    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Check if any material is Neo-Hookean
    use_newton = any(
        mat.get("type", "").upper() == "NEO_HOOKE"
        for mat in materials.values()
    )

    # -------------------------------------------------------------------------
    # Assemble global stiffness using element procedure
    # -------------------------------------------------------------------------
    for block in blocks:
        A = block["element"]["properties"]["area"]
        material = materials[block["material"]]

        for nodes in block["connect"]:
            # element nodal x-coordinates (length 2)
            x_e = coords[nodes, 0]

            # 2x2 element stiffness using Gauss quadrature
            ke = t1d1_element_stiffness(x_e, A, material, ngauss=2)

            # GLOBAL DOF = NODE NUMBER × DOF PER NODE + LOCAL DOF
            eft = [
                global_dof(n, j, dof_per_node)
                for n in nodes
                for j in range(dof_per_node)
            ]
            K[np.ix_(eft, eft)] += ke

    # -------------------------------------------------------------------------
    # Apply Neumann boundary conditions (concentrated nodal forces)
    # -------------------------------------------------------------------------
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                F[I] += bc["value"]

    # -------------------------------------------------------------------------
    # Apply distributed loads using element external-force procedure
    # (keeps the same qL/2 formula as the original implementation)
    # -------------------------------------------------------------------------
    for dload in dloads:
        dtype = dload["type"]
        direction = np.array(dload["direction"], dtype=float)

        if direction.size != 1:
            raise ValueError(
                f"1D problem expects one direction component, got {direction}"
            )

        sign = np.sign(direction[0])
        if sign == 0.0:
            raise ValueError(
                f"dload direction must be ±1, got {direction[0]}"
            )

        for eid in dload["elements"]:
            if eid not in block_elem_map:
                raise ValueError(
                    f"Element {eid} in distributed load "
                    f"{dload.get('name', '')!r} not found in any element block"
                )

            block_index, local_index = block_elem_map[eid]
            block = blocks[block_index]
            nodes = block["connect"][local_index]
            x_e = coords[nodes, 0]
            A = block["element"]["properties"]["area"]
            mat = materials[block["material"]]

            # Match original behaviour:
            #   BX:   q = value * sign               (line load)
            #   GRAV: q = rho * A * g * sign        (effective line load)
            if dtype == "BX":
                q = dload["value"] * sign
            elif dtype == "GRAV":
                rho = mat["density"]
                g = dload["value"]
                q = rho * A * g * sign
            else:
                raise NotImplementedError(
                    f"dload type {dtype!r} not supported for 1D"
                )

            # Original code used q * L/2 at each node.  Here we call the
            # element load routine with area = 1.0 and q as the effective
            # line load, so we recover the same result.
            fe = t1d1_element_uniform_load(x_e, area=1.0, q=q, direction=1.0)

            eft = [
                global_dof(n, j, dof_per_node)
                for n in nodes
                for j in range(dof_per_node)
            ]
            F[eft] += fe

    # -------------------------------------------------------------------------
    # Dirichlet boundary conditions via elimination (same as original)
    # -------------------------------------------------------------------------
    prescribed_dofs: list[int] = []
    prescribed_vals: list[float] = []

    for bc in bcs:
        if bc["type"] == DIRICHLET:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                prescribed_dofs.append(I)
                prescribed_vals.append(bc["value"])

    prescribed_dofs = np.array(prescribed_dofs, dtype=int)
    prescribed_vals = np.array(prescribed_vals, dtype=float)

    all_dofs = np.arange(num_dof, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # ---------------------------------------------------------------------
    # Linear solver (Week 1–2) if no neo-Hooke material present
    # ---------------------------------------------------------------------
    if not use_newton:

        Kff = K[np.ix_(free_dofs, free_dofs)]
        Kfp = K[np.ix_(free_dofs, prescribed_dofs)]
        Ff = F[free_dofs] - Kfp @ prescribed_vals

        uf = np.linalg.solve(Kff, Ff)

        dofs = np.zeros(num_dof, dtype=float)
        dofs[free_dofs] = uf
        dofs[prescribed_dofs] = prescribed_vals

        return {
            "dofs": dofs,
            "stiff": K,
            "force": F,
        }

    # ---------------------------------------------------------------------
    # Newton–Raphson nonlinear solver (Week 3)
    # ---------------------------------------------------------------------
    # Initial guess
    u = np.zeros(num_dof, dtype=float)
    u[prescribed_dofs] = prescribed_vals

    max_iter = 20
    tol = 1e-8

    for it in range(max_iter):

        # Build external forces
        Fext = assemble_external_forces(
            coords, blocks, bcs, dloads, materials, block_elem_map,
            num_dof, dof_per_node
        )

        # Build internal forces
        Fint = assemble_internal_forces(
            coords, blocks, materials, block_elem_map,
            u, dof_per_node
        )

        # Residual
        R = Fint - Fext

        # Check convergence (only free DOFs)
        Rf = R[free_dofs]
        if np.linalg.norm(Rf) < tol:
            break

        # Build tangent stiffness matrix at current displacement
        K = np.zeros((num_dof, num_dof))

        for block in blocks:
            A = block["element"]["properties"]["area"]
            material = materials[block["material"]]

            for nodes in block["connect"]:
                x_e = coords[nodes, 0]
                # tangent stiffness
                ke = t1d1_element_stiffness(x_e, A, material, ngauss=2)

                eft = [
                    global_dof(n, j, dof_per_node)
                    for n in nodes
                    for j in range(dof_per_node)
                ]
                K[np.ix_(eft, eft)] += ke

        # Extract free–free block
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Rf = R[free_dofs]

        # Newton step: solve Kff * du = -Rf
        du_f = np.linalg.solve(Kff, -Rf)

        # Update displacements
        u[free_dofs] += du_f

    dofs = u

    return {
        "dofs": dofs,
        "stiff": K,
        "force": -R,  # final applied force balance
    }


# -------------------------------------------------------------------------
# Helper: assemble internal force vector for given displacement u
# -------------------------------------------------------------------------
def assemble_internal_forces(
    coords, blocks, materials, block_elem_map, u, dof_per_node=1
):
    num_dof = len(u)
    fint = np.zeros(num_dof)
    for elem_id, (block_i, local_i) in block_elem_map.items():
        block = blocks[block_i]
        nodes = block["connect"][local_i]
        x_e = coords[nodes, 0]

        material = materials[block["material"]]
        A = block["element"]["properties"]["area"]

        u_e = np.array([u[nodes[0]], u[nodes[1]]])

        # Internal force from element routine
        from .elements import t1d1_element_internal_force
        f_int_e = t1d1_element_internal_force(x_e, u_e, A, material)

        # Assemble into global array
        for a, node_a in enumerate(nodes):
            ia = global_dof(node_a, 0, dof_per_node)
            fint[ia] += f_int_e[a]

    return fint


# -------------------------------------------------------------------------
# Helper: assemble external nodal forces (same as Week 2)
# -------------------------------------------------------------------------
def assemble_external_forces(
    coords, blocks, bcs, dloads, materials, block_elem_map, num_dof, dof_per_node=1
):
    Fext = np.zeros(num_dof)

    # Neumann BCs
    from .schemas import NEUMANN
    for bc in bcs:
        if bc["type"] == NEUMANN:
            for n in bc["nodes"]:
                I = global_dof(n, bc["local_dof"], dof_per_node)
                Fext[I] += bc["value"]

    # Distributed loads
    from .elements import t1d1_element_uniform_load
    for dload in dloads:
        direction = np.array(dload["direction"], float)
        sign = np.sign(direction[0])

        for elem_id in dload["elements"]:
            block_i, local_i = block_elem_map[elem_id]
            block = blocks[block_i]
            nodes = block["connect"][local_i]
            x_e = coords[nodes, 0]
            A = block["element"]["properties"]["area"]
            mat = materials[block["material"]]
            rho = mat.get("density", 0.0)

            if dload["type"] == "BX":
                q = dload["value"] * sign
            elif dload["type"] == "GRAV":
                g = dload["value"]
                q = rho * A * g * sign

            f_ext_e = t1d1_element_uniform_load(x_e, area=1.0, q=q, direction=1.0)

            for a, node_a in enumerate(nodes):
                ia = global_dof(node_a, 0, dof_per_node)
                Fext[ia] += f_ext_e[a]

    return Fext

def global_dof(node: int, local_dof: int, dof_per_node: int) -> int:
    """Return the global degree of freedom index for a given node and local dof."""
    return node * dof_per_node + local_dof
