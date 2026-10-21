"""Compatibilité du notebook 09 ; backend commun sous src/backends/."""
from pathlib import Path
import sys

# Conserver le lancement des anciens scripts depuis leur propre répertoire.
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.backends import local_llm_agent as _implementation
sys.modules[__name__] = _implementation
