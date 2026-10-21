"""Tester le backend avec processus simulés et les sorties historiques en lecture seule."""
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from src.agents import TrafficPredictionControlAgent
from src.agents.decision_agent import create_runtime
from src.backends import local_llm_agent
from src.backends.llama_cpp import LlamaCppBackend
from tests.multi_agent_integration_cases import run_integration_cases


ROOT = Path(__file__).resolve().parents[1]


class LlamaCppBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = LlamaCppBackend(ROOT)
        self.addCleanup(self.backend.close)

    def startup_mocks(self, stack):
        runtime = self.backend._runtime
        stack.enter_context(patch.object(runtime, "setup_saved_inputs"))
        stack.enter_context(patch.object(runtime.rules, "file_hash", return_value="fixture"))
        stack.enter_context(patch.object(runtime.psutil, "virtual_memory", return_value=MagicMock(available=2_000_000_000)))
        provenance = {"files": {"fixture.gguf": {"sha256": "fixture"}}, "llama_server_path": "tools/llama-server.exe"}
        stack.enter_context(patch.object(Path, "read_text", return_value=json.dumps(provenance)))
        channel = MagicMock()
        channel.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
        stack.enter_context(patch("src.backends.llama_cpp.socket.socket", return_value=channel))
        process = MagicMock()
        process.poll.return_value = None
        launch = stack.enter_context(patch("src.backends.llama_cpp.subprocess.Popen", return_value=process))
        response = MagicMock()
        response.__enter__.return_value.status = 200
        opener = MagicMock()
        opener.open.return_value = response
        stack.enter_context(patch("src.backends.llama_cpp.urllib.request.build_opener", return_value=opener))
        return launch, process, opener

    def test_start_call_and_cleanup(self):
        with ExitStack() as stack:
            launch, process, _ = self.startup_mocks(stack)
            with self.backend as backend:
                self.assertTrue(backend.available)
                log_path = Path(backend._temporary.name)
                expected = {"reason": "Fixture simulée"}
                def infer(state):
                    backend._runtime.LAST_CALL = {"raw_text": "fixture", "parsed_output": expected}
                    return expected
                with patch.object(backend._runtime, "llm_decision", side_effect=infer):
                    self.assertEqual(backend({}), expected)
                self.assertEqual(backend.last_call["parsed_output"], expected)
            self.assertFalse(log_path.exists())
            process.terminate.assert_called_once()
            self.assertFalse(self.backend.available)
            args = launch.call_args.args[0]
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertEqual(args[args.index("-c") + 1], "4096")
            self.assertEqual(args[args.index("-t") + 1], "4")

    def test_failed_loading_preserves_fallback_diagnostic(self):
        with patch.object(self.backend._runtime, "setup_saved_inputs", side_effect=FileNotFoundError("Fixture absente")), \
                patch("src.backends.llama_cpp.subprocess.Popen") as launch:
            with self.backend:
                self.assertFalse(self.backend.available)
                with self.assertRaisesRegex(RuntimeError, "LLM indisponible"):
                    self.backend({})
                self.assertIn("FileNotFoundError", self.backend.last_call["error"])
            launch.assert_not_called()

    def test_low_memory_does_not_start_process(self):
        with ExitStack() as stack:
            launch, _, _ = self.startup_mocks(stack)
            with patch.object(self.backend._runtime.psutil, "virtual_memory", return_value=MagicMock(available=999_999_999)):
                with self.backend:
                    self.assertIn("MemoryError", self.backend.load_error)
                launch.assert_not_called()

    def test_health_timeout_stops_process(self):
        with ExitStack() as stack:
            _, process, _ = self.startup_mocks(stack)
            stack.enter_context(patch("src.backends.llama_cpp.time.monotonic", side_effect=[0, 61]))
            with self.backend:
                self.assertIn("TimeoutError", self.backend.load_error)
                self.assertFalse(self.backend.available)
            process.terminate.assert_called()

    def test_exception_during_inference_closes_server(self):
        with ExitStack() as stack:
            _, process, _ = self.startup_mocks(stack)
            with self.assertRaises(TimeoutError):
                with self.backend:
                    with patch.object(self.backend._runtime, "llm_decision", side_effect=TimeoutError("Fixture")):
                        self.backend({})
            process.terminate.assert_called()

    def test_stop_kills_unresponsive_process(self):
        runtime = self.backend._runtime
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), None]
        runtime.SERVER = process
        runtime.SERVER_LOG = MagicMock()
        log = runtime.SERVER_LOG
        self.backend.close()
        process.kill.assert_called_once()
        log.close.assert_called_once()
        self.assertIsNone(runtime.SERVER)

    def test_runtime_and_legacy_import_compatibility(self):
        from scripts import local_llm_agent as legacy
        self.assertIs(legacy, local_llm_agent)
        other = LlamaCppBackend(ROOT)
        self.addCleanup(other.close)
        self.backend._runtime.rules.CONFIG["llm_model"] = "fixture"
        self.assertIsNone(other._runtime.rules.CONFIG["llm_model"])


class IntegrationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rules = create_runtime()
        with patch.object(rules, "inspect_environment", return_value={"selected_llm": None}), redirect_stdout(io.StringIO()):
            rules.initialize(ROOT)
        with patch("src.agents.traffic_prediction_agent.create_runtime", return_value=rules), \
                patch.object(rules, "initialize"):
            cls.agent = TrafficPredictionControlAgent(ROOT)

    def test_24_existing_cases_unchanged(self):
        self.assertIsNone(self.agent._init_error)
        _, actual = run_integration_cases(self.agent, ROOT)
        expected = json.loads((ROOT / "results/multi_agent_integration/integration_test_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(actual), 24)
        self.assertEqual(actual, expected)
        self.assertTrue(all(item["passed"] for item in actual))

    def test_saved_demo_response_and_diagnostics(self):
        notebook = json.loads((ROOT / "notebooks/10_multi_agent_integration.ipynb").read_text(encoding="utf-8"))
        def stream(index):
            return "".join("".join(o.get("text", [])) for o in notebook["cells"][index].get("outputs", []))
        raw = stream(20).split("Sortie LLM brute :\n", 1)[1].split("\nTemps d’inférence", 1)[0]
        output = json.loads(raw)
        expected = json.loads(stream(26))
        rules = self.agent.rules
        row_index = (rules.PREDICTIONS.predicted_occupancy_t15 - rules.PREDICTIONS.current_occupancy).idxmax()
        row = rules.PREDICTIONS.loc[row_index]
        state = rules.build_traffic_state(int(row.location), int(row.timestep))
        message = {"request_id": "demo_final_001", "timestamp": None,
                   "sensor_id": state["sensor_id"], "timestep": state["timestep"],
                   "traffic_data": {k: state[k] for k in ["current_flow", "current_speed", "current_occupancy"]},
                   "incident_context": {"incident_detected": None, "incident_type": None, "severity": None},
                   "emergency_context": {"emergency_vehicle_present": None, "priority_level": None},
                   "cooperative_context": {"neighboring_sensors": copy.deepcopy(state["neighbors"])},
                   "public_transport_context": {"priority_requested": None}, "integration_test_fixture": False}
        enriched = self.agent.build_state(message)
        backend = LlamaCppBackend(ROOT)
        self.addCleanup(backend.close)
        backend._runtime.rules.THRESHOLDS = rules.THRESHOLDS
        backend.last_call["parsed_output"] = output
        # Rejouer la sortie historique ; aucune génération de modèle n'est nécessaire.
        with patch.object(self.agent, "llm_backend", return_value=output):
            actual = self.agent.handle_message(message)
        self.assertEqual(actual, expected)
        report = backend.validation_report(self.agent, enriched)
        saved_table = next(o["data"]["text/html"] for o in notebook["cells"][22]["outputs"] if "text/html" in o.get("data", {}))
        # Lire le tableau sauvegardé avec la bibliothèque standard, sans dépendance HTML.
        class TableReader(HTMLParser):
            def __init__(self):
                super().__init__()
                self.cells = []
                self.collecting = False

            def handle_starttag(self, tag, attrs):
                if tag == "td":
                    self.collecting = True
                    self.cells.append("")

            def handle_data(self, data):
                if self.collecting:
                    self.cells[-1] += data

            def handle_endtag(self, tag):
                if tag == "td":
                    self.collecting = False

        reader = TableReader()
        reader.feed("".join(saved_table))
        self.assertEqual(report, {reader.cells[i].strip(): reader.cells[i + 1].strip() == "OK"
                                  for i in range(0, len(reader.cells), 2)})

        # Exécuter les cellules de démonstration refactorisées avec la sortie rejouée.
        from src.llm import LocalLLMBackend

        namespace = {}
        with patch("src.agents.traffic_prediction_agent.create_runtime", return_value=rules), \
                patch.object(rules, "initialize"), patch("IPython.display.display"), \
                redirect_stdout(io.StringIO()):
            exec("".join(notebook["cells"][14]["source"]), namespace)
            backend._runtime.rules.THRESHOLDS = rules.THRESHOLDS
            backend._runtime.LAST_CALL = {
                "raw_text": raw, "parsed_output": output, "error": None, "inference_time_s": 0.}
            def start(instance):
                instance._backend = backend
                backend.available = True
                return instance
            with patch.object(LocalLLMBackend, "start", start), \
                    patch.object(backend._runtime, "llm_decision", return_value=output):
                for index in [16, 18, 20, 22, 24, 26, 28]:
                    exec("".join(notebook["cells"][index]["source"]), namespace)
        self.assertEqual(namespace["demo_response"], expected)
        self.assertEqual(namespace["demo_checks"], report)


if __name__ == "__main__":
    unittest.main()
