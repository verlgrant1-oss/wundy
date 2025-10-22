import io

import numpy as np

import wundy


def test_load_input():
    file = io.StringIO()
    file.write("""\
wundy:
  coords: [0, 1, 2]
  connect: [[0, 1], [1, 2]]
""")
    file.seek(0)
    inp = wundy.load_input(file)
    assert np.allclose(inp["wundy"]["coords"], [0.0, 1.0, 2.0])
    assert isinstance(inp["wundy"]["coords"], np.ndarray)
    assert inp["wundy"]["coords"].dtype == float
    assert np.allclose(inp["wundy"]["connect"], [[0, 1], [1, 2]])
    assert isinstance(inp["wundy"]["connect"], np.ndarray)
    assert inp["wundy"]["connect"].dtype == int
