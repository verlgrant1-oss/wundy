import io

import numpy as np

import wundy


def test_load_input():
    file = io.StringIO()
    file.write(
        """\
wundy:
  coords: [0, 1, 2]
  connect: [[0, 1], [1, 2]]
  nset:
  - name: nset-1
    nodes: [1]
  boundary:
  - node: 0
    dof: 'x'
  - nset: 'nset-1'
    amplitude: 1.0
  cload:
  - node: 2
    amplitude: 2.0
  material:
  - type: elastic
    name: mat-1
    parameters:
      E: 10.0
      nu: 0.3
  element block:
  - material: mat-1
    name: block-1
    elements: all
    element_type: t1d1
"""
    )
    file.seek(0)
    data = wundy.ui.load(file)

    inp = data["wundy"]

    # Input file validator converted lists to arrays
    assert np.allclose(inp["coords"], [[0.0], [1.0], [2.0]])
    assert isinstance(inp["coords"], np.ndarray)
    assert inp["coords"].dtype == float

    assert np.allclose(inp["connect"], [[0, 1], [1, 2]])
    assert isinstance(inp["connect"], np.ndarray)
    assert inp["connect"].dtype == int

    # And inserts default values
    assert inp["boundary"] == [
        {"node": 0, "type": "dirichlet", "dof": 0, "amplitude": 0.0},
        {"nset": "nset-1", "type": "dirichlet", "dof": 0, "amplitude": 1.0},
    ]

    nsets = inp["nset"]
    assert isinstance(nsets, list)
    assert len(nsets) == 1
    assert np.allclose(nsets[0]["nodes"], [1])
    assert inp["cload"] == [{"node": 2, "dof": 0, "amplitude": 2.0}]

    materials = inp["material"]
    assert isinstance(materials, list)
    assert len(materials) == 1

    material = materials[0]
    assert material["type"] == "elastic"
    assert material["name"] == "mat-1"
    assert material["parameters"] == {"E": 10.0, "nu": 0.3}

    blocks = inp["element block"]
    assert isinstance(blocks, list)
    assert len(blocks) == 1

    block = blocks[0]
    assert block["element_type"] == "t1d1"
    assert block["name"] == "block-1"
    assert block["material"] == "mat-1"
    assert block["elements"] == "all"
    assert block["element_properties"] == {"area": 1.0}

    # preprocess will create doftags/dofvals/matprops
    d = wundy.ui.preprocess(data)

    assert np.allclose(d["doftags"], [[1], [1], [0]])
    assert np.allclose(d["dofvals"], [[0.0], [1.0], [2.0]])

    assert isinstance(d["element blocks"], dict)
    assert d["element blocks"] == {
        "block-1": {
            "material": "mat-1",
            "elements": [0, 1],
            "element_type": "t1d1",
            "element_properties": {"area": 1.0},
        }
    }
    assert np.allclose(d["dload"], np.zeros((2, 1)))
