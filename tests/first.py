import io

import wundy
import wundy.first


def test_first_1():
    file = io.StringIO()
    file.write("""\
wundy:
  coords: [0, 1, 2, 3, 4]
  connect: [[0, 1], [1, 2], [2, 3], [3, 4]]
  boundary:
  - node: 0
  cload:
  - node: 4
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
""")
    file.seek(0)
    data = wundy.ui.load(file)
    inp = wundy.ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["connect"],
        inp["doftags"],
        inp["dofvals"],
        inp["dload"],
        inp["materials"],
        inp["element blocks"],
    )

    u = soln["displ"]
    K = soln["stiff"]
    F = soln["force"]
    assert 0, "This test is not complete.  Are u, K, F correct?"


def test_first_2():
    assert 0, "This test is not implemented.  It should prescribe a distributed load"
