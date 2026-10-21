"""Exemples JSON recopiés des échanges historiques existants, sans nouvelle inférence."""
from importlib.resources import files
import json


EXAMPLE_NAMES = ("input", "output_warning", "output_error")


def load_example(name):
    """Renvoyer une copie indépendante d'un exemple livré avec le paquet."""
    if name not in EXAMPLE_NAMES:
        raise ValueError(f"Exemple inconnu : {name}. Valeurs possibles : {', '.join(EXAMPLE_NAMES)}.")
    resource = files("src.contracts").joinpath("example_json", f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))
