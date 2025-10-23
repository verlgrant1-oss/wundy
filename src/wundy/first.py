from typing import Any

import numpy as np
from numpy.typing import NDArray

from .schemas import DIRICHLET


def first_fe_code(
    coords: NDArray[float],
    connect: NDArray[int],
    doftags: NDArray[int],
    dofvals: NDArray[float],
    dload: NDArray[float],
    materials: dict[str, Any],
    blocks: dict[str, Any],
) -> dict[str, Any]:
    """
    Perform a single 1-D linear finite element analysis (axial bar, small strain).

    Parameters
    ----------
    coords : (nnode, 1) float array
        Nodal x-coordinates.
    connect : (nelem, 2) int array
        Element connectivity (2-node bars).
    doftags : (nnode, 1) int array
        DOF tags; DIRICHLET denotes prescribed displacement.
    dofvals : (nnode, 1) float array
        For DIRICHLET dofs: prescribed displacement value.
        For free dofs: may contain concentrated nodal force (cload) to add to F.
    dload : (nelem,) float array
        Uniform distributed load per element (force/length). May be zeros.
    materials : dict
        Materials with parameters (uses E for linear elastic).
    blocks : dict
        Element blocks with {"material": name, "element_properties": {"area": A}, "elements": [...]}

    Returns
    -------
    dict with:
      "displ" : (nnode,) float array, nodal displacements
      "K"     : (ndof, ndof) float array, global stiffness
      "F"     : (ndof,) float array, global load vector (including cload + distributed)
    """
    nnode, dof_per_node = coords.shape
    nelem, nper = connect.shape
    assert dof_per_node == 1, "Expect 1 DOF per node (axial u)."
    assert nper == 2, "Expect 2-node bar elements."

    ndof = nnode * dof_per_node
    K = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    # (A) Concentrated loads from dofvals ONLY on non-Dirichlet DOFs
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            if tag != DIRICHLET:
                I = n * dof_per_node + j
                F[I] += dofvals[n, j]

    # (B) Assemble element stiffness & consistent distributed load
    for block in blocks.values():
        A = float(block["element_properties"]["area"])
        mat = materials[block["material"]]
        E = float(mat["parameters"]["E"])
        for e in block["elements"]:
            nodes = connect[e]  # [i, j]
            dofs = [n * dof_per_node + j for n in nodes for j in range(dof_per_node)]
            xe = coords[nodes, 0]
            Le = float(xe[1] - xe[0])
            if np.isclose(Le, 0.0):
                raise ValueError(f"Zero-length element between nodes {nodes}")

            ke = (A * E / Le) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            K[np.ix_(dofs, dofs)] += ke

            q = float(dload[e]) if dload is not None and len(dload) > 0 else 0.0
            qe = (q * Le / 2.0) * np.ones(2)
            F[dofs] += qe

    # (C) Apply Dirichlet BCs (symmetric)
    Kbc = K.copy()
    Fbc = F.copy()
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            if tag == DIRICHLET:
                I = n * dof_per_node + j
                # Move known displacement to RHS, preserve symmetry
                Fbc -= Kbc[:, I] * dofvals[n, j]
                Kbc[I, :] = 0.0
                Kbc[:, I] = 0.0
                Kbc[I, I] = 1.0
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            if tag == DIRICHLET:
                I = n * dof_per_node + j
                Fbc[I] = dofvals[n, j]

    # (D) Solve
    u = np.linalg.solve(Kbc, Fbc)
    return {"displ": u, "K": K, "F": F}
