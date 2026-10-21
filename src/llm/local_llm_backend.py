"""Façade locale Qwen/llama.cpp réutilisant le transport et les prompts historiques."""
import copy
import json
from pathlib import Path


class LocalLLMBackend:
    """Backend injectable comme callable ou via generate.

    start est explicite et conserve les erreurs dans load_error sans les lever,
    comme le backend historique. generate lève si le backend est indisponible :
    le moteur applique alors son fallback habituel. stop libère les ressources.
    Aucun téléchargement ni démarrage implicite n'est effectué.
    """

    def __init__(self, project_root=None):
        self.root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        self._backend = None
        self._opened = False
        self.load_error = None
        self._last_call = {"raw_text": None, "parsed_output": None,
                           "error": None, "inference_time_s": None}

    @property
    def available(self):
        return self._backend is not None and self._backend.available

    @property
    def last_call(self):
        return copy.deepcopy(self._backend.last_call if self._backend is not None else self._last_call)

    def _check_local_files(self):
        # Vérifier les fichiers avant les diagnostics optionnels torch/transformers.
        folder = self.root / "models/qwen2.5-0.5b-instruct-gguf"
        model = folder / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        if not model.is_file():
            raise FileNotFoundError(f"Modèle GGUF Qwen absent : {model}")
        provenance_path = folder / "download_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Provenance locale llama.cpp absente : {provenance_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        executable = self.root / provenance["llama_server_path"]
        if not executable.is_file():
            raise FileNotFoundError(f"Exécutable llama-server absent : {executable}")
        # Le transport historique conserve ses vérifications SHA-256, dont release.zip.

    def start(self):
        """Ouvrir explicitement une session ; consulter available et load_error."""
        if self._opened:
            raise RuntimeError("Ce backend est déjà ouvert.")
        self._opened = True
        self._backend = None
        self.load_error = None
        self._last_call = {"raw_text": None, "parsed_output": None,
                           "error": None, "inference_time_s": None}
        try:
            self._check_local_files()
            # Import différé : même la construction ne charge pas le transport historique.
            from src.backends.llama_cpp import LlamaCppBackend

            self._backend = LlamaCppBackend(self.root)
            self._backend.__enter__()
            self.load_error = self._backend.load_error
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
            if self._backend is not None:
                self._backend.close()
        return self

    def generate(self, state):
        """Générer une proposition ; les validations métier restent dans guardrails."""
        if not self.available:
            error = self.load_error or "Backend non démarré ; appeler start() explicitement."
            self._last_call["error"] = error
            if self._backend is not None:
                self._backend.last_call["error"] = error
            raise RuntimeError("LLM indisponible : " + error)
        # Aucun changement de prompt, parsing, délai, schéma ou paramètres de génération.
        return self._backend(state)

    def stop(self):
        """Arrêter le serveur et nettoyer les journaux temporaires ; appel idempotent."""
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._opened = False

    def __call__(self, state):
        return self.generate(state)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def validation_report(self, agent, state):
        """Conserver le rapport qualitatif existant après un appel au transport."""
        if self._backend is None:
            raise RuntimeError("Aucun transport initialisé pour le rapport de validation.")
        return self._backend.validation_report(agent, state)
