import yaml

from .schemas import input_schema


def load_input(file):
    data = yaml.safe_load(file)
    return input_schema.validate(data)
