import toml
from munch import Munch
import json


def load_toml(path: str):
    with open(path, mode="r")as f:
        toml_data = toml.load(f)
    return Munch.fromDict(toml_data)


def load_json(path: str):
    with open(path, mode="r")as f:
        json_data = json.load(f)
    return Munch.fromDict(json_data)
