"""Ancien chemin compatible ; politique canonique dans src.decision.policy."""
import sys
from src.decision import policy as _implementation

# Préserver l'identité du module et les anciens imports, sans copie de valeurs.
sys.modules[__name__] = _implementation
