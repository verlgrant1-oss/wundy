from typing import Any

import numpy as np
from numpy.typing import NDArray


NEUMANN = 0
DIRICHLET = 1

def first_fe_code(
    coords: NDArray[float],
    connect: NDArray[int],
    doftags: NDArray[int],
    dofvals: NDArray[float],
    dload: NDArray[float],
    materials: dict[str, Any],
    blocks: dict[str, Any],
) -> dict[str, Any]:
    num_node, dof_per_node = coords.shape
    num_elem, node_per_elem = connect.shape

    num_dof = num_node * dof_per_node
    K = np.zeros((num_dof, num_dof), dtype=float)
    F = np.zeros(num_dof, dtype=float)

    # Assign concentrated loads (Neumann BC to F)
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            # n is the node number, j is the local dof
            if tag == 0:
                I = n * dof_per_node + j
                F[I] = dofvals[n, j]

    # Assemble global stiffness
    for block in blocks.values():
        A = block["element_properties"]["area"]
        material = materials[block["material"]]
        E = material["parameters"]["E"]
        for element in block["elements"]:
            nodes = connect[element]

            # GLOBAL DOF = NODE NUMBER x NUMBER OF DOF PER NODE + LOCAL DOF
            dofs = [n * dof_per_node + j for n in nodes for j in range(dof_per_node)]

            xe = coords[nodes]
            he = xe[1, 0] - xe[0, 0]
            ke = A * E / he * np.array([[1.0, -1.0], [-1.0, 1.0]])
            K[np.ix_(dofs, dofs)] += ke

            # Distributed load contribution
            qe = dload[element] * he / 2.0 * np.ones(2)
            F[np.ix_(dofs)] += qe

    # Apply boundary conditions
    Kbc = K.copy()
    Fbc = F.copy()
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            # n is the node number, j is the local dof
            if tag == DIRICHLET:
                # Dirichlet
                # Apply boundary conditions such that the matrix remains symmetric
                I = n * dof_per_node + j
                Fbc -= K[:, I] * dofvals[n, j]
                Kbc[I, :] = Kbc[:, I] = 0
                Kbc[I, I] = 1

    # Further modify RHS for Dirichlet boundary
    # This must be done after the loop above.
    for n, tags in enumerate(doftags):
        for j, tag in enumerate(tags):
            if tag == DIRICHLET:
                I = n * dof_per_node + j
                Fbc[I] = dofvals[n, j]

    # solve the system
    u = np.linalg.solve(Kbc, Fbc)

    solution = {"displ": u, "stiff": K, "force": F}

    return solution
