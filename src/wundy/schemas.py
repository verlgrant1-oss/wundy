import numpy as np
import schema
from schema import And, Use

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
        }
    }
)
