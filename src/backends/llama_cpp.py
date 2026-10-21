"""Backend callable llama.cpp local ; cycle de vie et journaux temporaires isolés."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

from . import local_llm_agent
from src.decision import guardrails


class LlamaCppBackend:
    """Charger uniquement les fichiers locaux et fermer le serveur en sortie de contexte.

    Un chargement refusé conserve son diagnostic. L'appel lève alors une erreur
    pour que TrafficPredictionControlAgent applique son fallback habituel.
    Aucun fichier du dépôt n'est écrit par ce backend.
    """

    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        spec = importlib.util.spec_from_file_location("_goflow_local_llm_runtime", local_llm_agent.__file__)
        self._runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._runtime)
        self._runtime.ROOT = self.root
        self.available = False
        self.load_error = None
        self.last_call = {"raw_text": None, "parsed_output": None,
                          "error": None, "inference_time_s": None}
        self._temporary = None

    def __enter__(self):
        if self._temporary is not None:
            raise RuntimeError("Ce backend est déjà ouvert.")
        self.last_call = {"raw_text": None, "parsed_output": None,
                          "error": None, "inference_time_s": None}
        self.load_error = None
        self._temporary = tempfile.TemporaryDirectory(prefix="goflow_final_demo_")
        runtime = self._runtime
        try:
            runtime.setup_saved_inputs()
            provenance_path = self.root / "models/qwen2.5-0.5b-instruct-gguf/download_provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            for relative, entry in provenance["files"].items():
                assert runtime.rules.file_hash(self.root / relative) == entry["sha256"]
            if runtime.psutil.virtual_memory().available < 1_000_000_000:
                raise MemoryError("Moins de 1 Go libre pour le backend local.")
            executable = self.root / provenance["llama_server_path"]
            model_path = self.root / "models/qwen2.5-0.5b-instruct-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            with socket.socket() as channel:
                channel.bind(("127.0.0.1", 0))
                port = channel.getsockname()[1]
            runtime.SERVER_URL = f"http://127.0.0.1:{port}"
            log_path = Path(self._temporary.name) / "llama_server.log"
            runtime.SERVER_LOG = log_path.open("w", encoding="utf-8")
            runtime.SERVER = subprocess.Popen(
                [str(executable), "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port),
                 "-t", "4", "-c", "4096", "-ngl", "0", "--parallel", "1"],
                cwd=executable.parent, stdout=runtime.SERVER_LOG, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if runtime.SERVER.poll() is not None:
                    raise RuntimeError("Serveur arrêté : " + log_path.read_text(
                        encoding="utf-8", errors="replace")[-1000:])
                try:
                    with opener.open(runtime.SERVER_URL + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                raise TimeoutError("Le modèle ne s’est pas chargé dans le délai prévu.")
            runtime.MODEL = "llama.cpp_server_ready"
            self.available = True
        except Exception as error:
            self.load_error = f"{type(error).__name__}: {error}"
            runtime.stop_model()
        return self

    def __call__(self, state):
        if not self.available:
            self.last_call["error"] = self.load_error
            raise RuntimeError("LLM indisponible : " + str(self.load_error))
        try:
            return self._runtime.llm_decision(state)
        finally:
            self.last_call.update(copy.deepcopy(self._runtime.LAST_CALL))

    def close(self):
        try:
            self._runtime.stop_model()
        finally:
            self.available = False
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def validation_report(self, agent, state):
        return guardrails.validation_report(
            self.last_call["parsed_output"], state, agent=agent, rules=self._runtime.rules)
