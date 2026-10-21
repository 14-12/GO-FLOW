"""Interface LLM publique : processus et HTTP simulés, aucun poids téléchargé."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from src.llm import LocalLLMBackend
from src.backends.llama_cpp import LlamaCppBackend
from tests import test_decision_layer as decision_cases

ROOT = Path(__file__).resolve().parents[1]


def test_import_et_construction_sans_effet_de_bord():
    code = """
import sys
from unittest.mock import patch
with patch('subprocess.Popen', side_effect=AssertionError('subprocess interdit')), patch('urllib.request.urlopen', side_effect=AssertionError('réseau interdit')):
    from src.llm import LocalLLMBackend
    backend = LocalLLMBackend()
    assert not backend.available
    assert 'src.backends.llama_cpp' not in sys.modules
    assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)


def test_backend_disponible_cycle_de_vie_et_injection():
    transport = MagicMock()
    transport.available = True
    transport.load_error = None
    transport.return_value = {"reason": "Proposition simulée"}
    backend = LocalLLMBackend(ROOT)
    with patch.object(backend, "_check_local_files"), patch("src.backends.llama_cpp.LlamaCppBackend", return_value=transport):
        with backend as opened:
            assert opened is backend
            assert backend.available
            assert backend.generate({}) == transport.return_value
            assert backend({}) == transport.return_value
            with pytest.raises(RuntimeError, match="déjà ouvert"):
                backend.start()
        transport.__enter__.assert_called_once()
        transport.close.assert_called_once()


@pytest.mark.parametrize("missing", ["gguf", "provenance", "llama-server"])
def test_fichiers_absents_erreur_claire_sans_processus(tmp_path, missing):
    folder = tmp_path / "models/qwen2.5-0.5b-instruct-gguf"
    folder.mkdir(parents=True)
    if missing != "gguf":
        (folder / "qwen2.5-0.5b-instruct-q4_k_m.gguf").touch()
    if missing == "llama-server":
        (folder / "download_provenance.json").write_text(json.dumps({"llama_server_path": "tools/llama-server.exe"}), encoding="utf-8")
    backend = LocalLLMBackend(tmp_path)
    with patch("subprocess.Popen") as launch:
        with backend:
            assert not backend.available
            assert "FileNotFoundError" in backend.load_error
            assert {"gguf": "GGUF", "provenance": "Provenance", "llama-server": "llama-server"}[missing] in backend.load_error
            with pytest.raises(RuntimeError, match="LLM indisponible"):
                backend.generate({})
        launch.assert_not_called()


def test_generation_ne_demarre_pas_implicitement():
    backend = LocalLLMBackend(ROOT)
    with patch.object(backend, "start") as start:
        with pytest.raises(RuntimeError, match=r"start\(\)"):
            backend.generate({})
        start.assert_not_called()
    backend.stop()
    backend.stop()


@pytest.fixture
def generation():
    case = decision_cases.DecisionLayerTests()
    case.setUp()
    transport = LlamaCppBackend(ROOT)
    transport.available = True
    transport._runtime.MODEL = "llama.cpp_server_ready"
    transport._runtime.SERVER_URL = "http://127.0.0.1:12345"
    transport._runtime.rules.THRESHOLDS = copy.deepcopy(case.thresholds)
    backend = LocalLLMBackend(ROOT)
    backend._backend = transport
    case.agent.llm_backend = backend
    yield backend, transport, case
    backend.stop()


@pytest.mark.parametrize("kind", ["json_valide", "json_invalide", "timeout", "erreur_serveur"])
def test_generation_http_et_fallback(generation, kind):
    backend, transport, case = generation
    expected = case.agent._rule_decision(case.state)
    opener = MagicMock()
    if kind in {"json_valide", "json_invalide"}:
        content = json.dumps(expected) if kind == "json_valide" else "{"
        answer = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        opener.open.return_value.__enter__.return_value = io.StringIO(json.dumps(answer))
    elif kind == "timeout":
        opener.open.side_effect = TimeoutError("Délai simulé")
    else:
        opener.open.side_effect = HTTPError("http://127.0.0.1:12345", 500, "Erreur simulée", {}, None)
    with patch("src.backends.local_llm_agent.urllib.request.build_opener", return_value=opener):
        result = case.agent.decide(copy.deepcopy(case.state))
    assert result["decision"] == expected
    if kind == "json_valide":
        assert result["decision_source"] == "llm"
        assert backend.last_call["parsed_output"] == expected
    else:
        assert result["decision_source"] == "rule_based_fallback"
        assert result["trace"]["fallback_reason"] == "llm_backend_error"
        assert backend.last_call["error"]
    # Vérifier les paramètres du vrai chemin HTTP sans effectuer de requête.
    request = opener.open.call_args.args[0]
    payload = json.loads(request.data)
    assert payload["messages"][0]["content"] == transport._runtime.SYSTEM_PROMPT
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 300
    assert opener.open.call_args.kwargs["timeout"] == transport._runtime.TIMEOUT_S


def test_indisponibilite_declenche_le_fallback_actuel():
    case = decision_cases.DecisionLayerTests()
    case.setUp()
    backend = LocalLLMBackend(ROOT)
    transport = MagicMock(available=False, load_error="TimeoutError: serveur indisponible")
    with patch.object(backend, "_check_local_files"), patch("src.backends.llama_cpp.LlamaCppBackend", return_value=transport):
        with backend:
            assert not backend.available
            assert "TimeoutError" in backend.load_error
            case.agent.llm_backend = backend
            result = case.agent.decide(copy.deepcopy(case.state))
    assert result["decision"] == case.agent._rule_decision(case.state)
    assert result["decision_source"] == "rule_based_fallback"
    assert result["trace"]["fallback_reason"] == "llm_backend_error"


def test_exception_dans_contexte_arrete_le_backend():
    backend = LocalLLMBackend(ROOT)
    transport = MagicMock(available=True, load_error=None)
    with patch.object(backend, "_check_local_files"), patch("src.backends.llama_cpp.LlamaCppBackend", return_value=transport):
        with pytest.raises(TimeoutError):
            with backend:
                raise TimeoutError("Erreur de l'appelant simulée")
    transport.close.assert_called_once()
