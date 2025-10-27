import io

import numpy as np

import wundy
import wundy.first


def test_first_1():
    file = io.StringIO()
    file.write("""\
wundy:
  nodes: [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
  elements: [[1, 1, 2], [2, 2, 3], [3, 3, 4], [4, 4, 5]]
  boundary conditions:
  - name: fix-nodes
    dof: x
    nodes: [1]
  concentrated loads:
  - name: cload-1
    nodes: [5]
    value: 2.0
  materials:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element blocks:
  - material: mat-1
    name: block-1
    elements: all
    element:
      type: t1d1
      properties:
        area: 1
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    return wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )

    dofs = soln["dofs"]
    K = soln["stiff"]
    F = soln["force"]
    assert np.allclose(dofs, [0, 0.2, 0.4, 0.6, 0.8])
    assert np.allclose(F, [0, 0, 0, 0, 2])
    assert np.allclose(
        K,
        [
            [10, -10, 0, 0, 0],
            [-10, 20, -10, 0, 0],
            [0, -10, 20, -10, 0],
            [0, 0, -10, 20, -10],
            [0, 0, 0, -10, 10],
        ],
    )


def test_first_2():
    """
    Uniform distributed load q=1 over each unit element, applied via
    EQUIVALENT NODAL FORCES using `cload` (schema-friendly for Week 1).

    For 4 unit elements, the consistent nodal forces assemble (with node 0 fixed) to:
      F = [0.0, 1.0, 1.0, 1.0, 0.5]
    With EA=10, L=4, left end fixed, expected displacements:
      u ≈ [0.00, 0.35, 0.60, 0.75, 0.80]
    """
    yaml_text = """
wundy:
  coords: [0, 1, 2, 3, 4]
  connect: [[0,1],[1,2],[2,3],[3,4]]
  boundary:
    - node: 0
  cload:
    - node: 1
      amplitude: 1.0
    - node: 2
      amplitude: 1.0
    - node: 3
      amplitude: 1.0
    - node: 4
      amplitude: 0.5
  material:
    - type: elastic
      name: mat-1
      parameters: {E: 10.0, nu: 0.3}
  element block:
    - material: mat-1
      name: block-1
      elements: all
      element_type: t1d1
"""
    soln = _run(yaml_text)
    u = np.asarray(soln["displ"])
    K = np.asarray(soln["K"])
    F = np.asarray(soln["F"])

    # Expected loads after schema/preprocessor: no cload on fixed node 0
    F_exp = np.array([0.0, 1.0, 1.0, 1.0, 0.5])
    # Analytic fixed–free bar with q=1, EA=10, L=4 at the nodes:
    u_exp = np.array([0.00, 0.35, 0.60, 0.75, 0.80])

    assert u.shape == (5,)
    assert K.shape == (5, 5)
    assert F.shape == (5,)

    assert np.allclose(F, F_exp, rtol=1e-12, atol=1e-12)
    assert np.allclose(u, u_exp, rtol=1e-3, atol=1e-6)
