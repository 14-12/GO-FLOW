"""Compatibilité des anciens imports ; implémentation commune sous src/agents/."""
from pathlib import Path
import sys

# Les scripts lancés directement doivent aussi trouver le paquet src du projet.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.agents import decision_agent as _implementation

# Conserver aussi les affectations historiques (THRESHOLDS, CONFIG, backend LLM).
sys.modules[__name__] = _implementation
