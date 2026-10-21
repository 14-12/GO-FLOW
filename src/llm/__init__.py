"""Interface LLM locale, sans chargement de modèle ni processus à l'import."""
from .local_llm_backend import LocalLLMBackend

__all__ = ["LocalLLMBackend"]
