import io

import numpy as np

import wundy


def test_load_input():
    file = io.StringIO()
    file.write("""\
wundy:
  coords: [0, 1, 2]
  connect: [[0, 1], [1, 2]]
  boundary:
  - node: 0
    dof: 'x'
  - node: 1
    amplitude: 1.0
  cload:
  - node: 2
    amplitude: 1.0
  material:
  - type: elastic
    parameters:
      E: 10.0
      nu: 0.3
    elements: [0, 1]
""")
    file.seek(0)
    inp = wundy.load_input(file)
    assert np.allclose(inp["wundy"]["coords"], [0.0, 1.0, 2.0])
    assert isinstance(inp["wundy"]["coords"], np.ndarray)
    assert inp["wundy"]["coords"].dtype == float
    assert np.allclose(inp["wundy"]["connect"], [[0, 1], [1, 2]])
    assert isinstance(inp["wundy"]["connect"], np.ndarray)
    assert inp["wundy"]["connect"].dtype == int
    assert inp["wundy"]["boundary"] == [
        {"node": 0, "type": "dirichlet", "dof": 0, "amplitude": 0.0},
        {"node": 1, "type": "dirichlet", "dof": 0, "amplitude": 1.0},
    ]
    assert inp["wundy"]["cload"] == [{"node": 2, "dof": 0, "amplitude": 1.0}]
    materials = inp["wundy"]["material"]
    assert isinstance(materials, list)
    assert len(materials) == 1
    material = materials[0]
    assert material["type"] == "elastic"
    assert np.allclose(material["elements"], [0, 1])
    assert isinstance(material["elements"], np.ndarray)
    assert material["elements"].dtype == int
    assert material["elements"].dtype == int
    assert material["parameters"] == {"E": 10.0, "nu": 0.3}
