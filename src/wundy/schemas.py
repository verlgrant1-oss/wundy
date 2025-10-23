from typing import Any

import numpy as np
from schema import And
from schema import Optional
from schema import Or
from schema import Schema
from schema import Use


def validate_material_parameters(material: dict[str, dict[str, Any]]) -> bool:
    elastic = Schema(
        {
            "E": And(float, lambda x: x > 0.0, error="E must be > 0"),
            "nu": And(float, lambda x: -1.0 <= x < 0.5, error="nu must be between -1 and .5"),
        }
    )
    if material["type"] == "elastic":
        elastic.validate(material["parameters"])
    else:
        raise ValueError(f"Unknown material {material['type']!r}")
    return True


def validate_element_properties(block: dict[str, dict[str, Any]]) -> bool:
    t1d1 = Schema({Optional("area", default=1.0): And(float, lambda a: a > 0)})
    if block["element type"].lower() == "t1d1":
        v = t1d1.validate(block["properties"])
        block["properties"].update(v)
    else:
        raise ValueError(f"Unknown element type {block['element type']!r}")
    return True


coords_schema = Schema(
    And(
        list,
        lambda f: all(isinstance(n, (float, int)) for n in f),
        Use(lambda x: np.array([[_] for _ in x], dtype=float)),
    )
)
connect_schema = Schema(
    And(
        list,
        lambda outer: all(isinstance(inner, list) for inner in outer),
        lambda outer: all(isinstance(n, int) for inner in outer for n in inner),
        Use(lambda x: np.array(x, dtype=int)),
    )
)

nset_schema = Schema(
    {
        "name": And(str, Use(lambda s: s.lower())),
        "nodes": And(
            list,
            lambda f: all(isinstance(n, int) for n in f),
            Use(lambda x: np.array(x, dtype=int)),
        ),
    },
)

elset_schema = Schema(
    {
        "name": And(str, Use(lambda s: s.lower())),
        "elements": And(
            list,
            lambda f: all(isinstance(n, int) for n in f),
            Use(lambda x: np.array(x, dtype=int)),
        ),
    },
)

boundary_schema = Schema(
    And(
        {
            Optional("amplitude", default=0.0): Use(float),
            Optional("type", default="dirichlet"): And(
                str,
                lambda s: s.lower() in ("dirichlet", "neumann"),
                Use(lambda s: s.lower()),
            ),
            Optional("dof", default=0): And(
                str,
                lambda s: s.lower() in "x",  # extension to 2/3D: allow dof to be xyz
                Use(lambda x: {"x": 0, "y": 1, "z": 2}[x.lower()]),
            ),
            Or("node", "nset"): object,
        },
        lambda d: ("node" in d) ^ ("nset" in d),
        lambda d: not ("node" in d and "nset" in d),
        lambda d: isinstance(d.get("node"), int) if "node" in d else True,
        lambda d: isinstance(d.get("nset"), str) if "nset" in d else True,
    )
)
cload_schema = Schema(
    And(
        {
            Optional("amplitude", default=0.0): Use(float),
            Optional("dof", default=0): And(
                str,
                lambda s: s.lower() in "x",  # extension to 2/3D: allow dof to be xyz
                Use(lambda x: {"x": 0, "y": 1, "z": 2}[x.lower()]),
            ),
            Or("node", "nset"): object,
        },
        lambda d: ("node" in d) ^ ("nset" in d),
        lambda d: not ("node" in d and "nset" in d),
        lambda d: isinstance(d.get("node"), int) if "node" in d else True,
        lambda d: isinstance(d.get("nset"), str) if "nset" in d else True,
    )
)
dload_schema = Schema(
    And(
        {
            Optional("amplitude", default=0.0): Use(float),
            Optional("dof", default=0): And(
                str,
                lambda s: s.lower() in "x",  # extension to 2/3D: allow dof to be xyz
                Use(lambda x: {"x": 0, "y": 1, "z": 2}[x.lower()]),
            ),
            Or("element", "elset"): object,
        },
        lambda d: ("element" in d) ^ ("elset" in d),
        lambda d: not ("element" in d and "elset" in d),
        lambda d: isinstance(d.get("element"), int) if "element" in d else True,
        lambda d: isinstance(d.get("elset"), str) if "elset" in d else True,
    )
)

material_schema = Schema(
    And(
        {
            "type": And(str, Use(lambda s: s.lower())),
            "name": And(str, Use(lambda s: s.lower())),
            "parameters": {str: object},
        },
        lambda d: validate_material_parameters(d),
    )
)
block_schema = Schema(
    And(
        {
            "name": And(str, Use(lambda s: s.lower())),
            "material": And(str, Use(lambda s: s.lower())),
            "elements": Or(
                And(str, Use(lambda s: s.lower())),
                And(list, lambda outer: all(isinstance(_, int) for _ in outer)),
            ),
            "element type": And(str, lambda s: s.lower() in ("t1d1",), Use(lambda n: n.lower())),
            Optional("properties", default=dict()): dict,
        },
        lambda d: validate_element_properties(d),
    )
)
input_schema = Schema(
    {
        "wundy": {
            "coords": coords_schema,
            "connect": connect_schema,
            Optional("nset"): [nset_schema],
            Optional("elset"): [elset_schema],
            "boundary": [boundary_schema],
            Optional("cload"): [cload_schema],
            Optional("dload"): [dload_schema],
            "material": [material_schema],
            "element block": [block_schema],
        }
    }
)
