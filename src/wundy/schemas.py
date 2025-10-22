import numpy as np
import schema
from schema import And
from schema import Optional
from schema import Or
from schema import Use

input_schema = schema.Schema(
    {
        "wundy": {
            "coords": And(
                list,
                lambda f: all(isinstance(n, (float, int)) for n in f),
                Use(lambda x: np.array(x, dtype=float)),
            ),
            "connect": And(
                list,
                lambda outer: all(isinstance(inner, list) for inner in outer),
                lambda outer: all(isinstance(n, int) for inner in outer for n in inner),
                Use(lambda x: np.array(x, dtype=int)),
            ),
            "boundary": [
                {
                    "node": int,
                    Optional("type", default="dirichlet"): And(
                        str, lambda s: s.lower() in ("dirichlet",), Use(lambda s: s.lower()),
                    ),
                    Optional("dof", default=0): And(
                        str,
                        lambda s: s.lower() in "x",
                        Use(lambda x: {"x": 0, "y": 1, "z": 2}[x.lower()])
                    ),
                    Optional("amplitude", default=0.0): float,
                }
            ],
            "cload": [  # concentrated loads
                {
                    "node": int,
                    Optional("dof", default=0): And(
                        str,
                        lambda s: s.lower() in "x",
                        Use(lambda x: {"x": 0, "y": 1, "z": 2}[x.lower()])
                    ),
                    Optional("amplitude", default=0.0): float,
                }
            ],
            "material": [
                {
                    "type": str,
                    "elements": And(
                        list,
                        lambda f: all(isinstance(n, int) for n in f),
                        Use(lambda x: np.array(x, dtype=int)),
                    ),
                    "parameters": {str: Use(float)}
                }
            ]
        }
    }
)
