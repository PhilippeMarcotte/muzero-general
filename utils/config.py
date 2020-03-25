import toml
from munch import Munch


def load_toml(path: str):
    with open(path, mode="r")as f:
        toml_data = toml.load(f)
    return Munch.fromDict(toml_data)
