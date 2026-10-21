"""Intégrer un vrai petit LLM CPU aux cinq états GoFlow déjà sauvegardés."""
from pathlib import Path
import copy
import gc
import importlib.util
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from time import perf_counter

import pandas as pd
import psutil

from src.agents.decision_agent import create_runtime
from src.decision import decision_engine, guardrails, policy

ROOT = Path(__file__).resolve().parents[2]
rules = create_runtime()

MODEL = None
TOKENIZER = None
SERVER = None
SERVER_URL = None
SERVER_LOG = None
LAST_CALL = {}
TIMEOUT_S = 60
SYSTEM_PROMPT = """Tu es l'agent Traffic Prediction & Adaptive Signal Control de GoFlow.
Tu reçois uniquement des données réelles déjà calculées. XGBoost prédit l'occupation
à +15 minutes : tu ne fais jamais de forecasting et ne modifies aucune prévision.
Analyse le trafic et le risque, choisis UNE action autorisée et justifie-la brièvement.
N'invente aucune mesure ni voisin. incident_detected=null et emergency_vehicle_present=null
signifient INCONNUS, pas absents. Ne détecte ni accident ni urgence. Pas de secondes de feu.
Pas de modification de phase car les plans de feux sont inconnus.
Réponds uniquement en JSON, exactement les clés traffic_state, risk_level,
recommended_action, reason, confidence. confidence est un float entre 0.0 et 1.0,
ton auto-évaluation non calibrée. Respecte les enums et garde-fous fournis.
Dans reason, cite l'occupation actuelle et la prévision avec leurs nombres fournis
à six décimales. Termine exactement par : Incidents inconnus; urgences inconnues;
plan de feux inconnu. Évite d'autres chiffres et ne fais pas d'affirmation sur des voisins.
"""


def setup_saved_inputs():
    import torch
    global INPUTS, SCENARIOS, PREVIOUS_OUTPUTS, MODEL_INFO
    folder = ROOT / "results/decision_agent"
    INPUTS = json.loads((folder / "agent_inputs.json").read_text(encoding="utf-8"))
    SCENARIOS = pd.read_csv(folder / "selected_scenarios.csv")
    PREVIOUS_OUTPUTS = json.loads((folder / "agent_outputs.json").read_text(encoding="utf-8"))
    previous_validation = pd.read_csv(folder / "validation_results.csv")
    method = (folder / "decision_agent_methodology.md").read_text(encoding="utf-8")
    record = json.loads(method.split("```json\n", 1)[1].split("\n```", 1)[0])
    rules.THRESHOLDS = record["thresholds_train_only"]
    rules.CONFIG.update(record["config"])
    assert len(INPUTS) == len(SCENARIOS) == len(PREVIOUS_OUTPUTS) == len(previous_validation) == 5
    for relative, checksum in record["input_hashes"].items():
        assert rules.file_hash(ROOT / relative) == checksum, f"Source modifiée : {relative}"
    for index, entry in enumerate(INPUTS):
        state = entry["state"]
        row = SCENARIOS.iloc[index]
        assert entry["scenario"] == row.scenario == PREVIOUS_OUTPUTS[index]["scenario"]
        assert state["sensor_id"] == int(row.location) and state["timestep"] == int(row.timestep)
        assert state["incident_detected"] is None and state["emergency_vehicle_present"] is None
        assert "target_occupy_t15" not in state
        assert rules.rule_based_decision(state) == PREVIOUS_OUTPUTS[index]["decision"]
    memory = psutil.virtual_memory()
    MODEL_INFO = {"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "device": "cpu",
                  "python_version": sys.version, "torch_version": torch.__version__,
                  "cuda_available": torch.cuda.is_available(),
                  "usable_gpu_count": torch.cuda.device_count(),
                  "ollama_executable": shutil.which("ollama"),
                  "llama_cpp_model_files": [str(p.relative_to(ROOT)) for p in (ROOT / "models").rglob("*.gguf")],
                  "ram_total_bytes": memory.total, "ram_available_before_load_bytes": memory.available,
                  "packages": {name: importlib.util.find_spec(name) is not None for name in
                               ["transformers", "torch", "llama_cpp", "ollama", "accelerate", "bitsandbytes"]},
                  "fine_tuned_on_pems08": False, "scenario_count": 5,
                  "source_inputs_sha256": rules.file_hash(folder / "agent_inputs.json"),
                  "soft_timeout_s": TIMEOUT_S, "max_new_tokens": 300,
                  "generation": "greedy, do_sample=False", "load_error": None}
    import transformers
    MODEL_INFO["transformers_version"] = transformers.__version__
    return MODEL_INFO


def load_model_transformers_diagnostic():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    global MODEL, TOKENIZER
    folder = ROOT / "models/qwen2.5-0.5b-instruct"
    process = psutil.Process()
    before = process.memory_info().rss
    start = perf_counter()
    try:
        if psutil.virtual_memory().available < 2_300_000_000:
            raise MemoryError("RAM libre insuffisante pour le pic de chargement float32 (2,3 Go requis).")
        provenance = json.loads((folder / "download_provenance.json").read_text(encoding="utf-8"))
        assert provenance["model_id"] == MODEL_INFO["model_id"]
        for filename, entry in provenance["files"].items():
            assert rules.file_hash(folder / filename) == entry["sha256"]
        torch.set_num_threads(4)
        torch.manual_seed(42)
        TOKENIZER = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        MODEL = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32,
                                                    attn_implementation="eager", local_files_only=True)
        MODEL.eval()
        MODEL_INFO["parameters_before_quantization"] = sum(p.numel() for p in MODEL.parameters())
        # Quantifier le backbone ; laisser embeddings et tête de sortie partagée en float32.
        torch.ao.quantization.quantize_dynamic(MODEL.model, {torch.nn.Linear},
                                               dtype=torch.qint8, inplace=True)
        MODEL_INFO["quantization"] = "INT8 dynamique des Linear du backbone ; embeddings/lm_head float32"
        MODEL_INFO["quantized_linear_count"] = sum(
            isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in MODEL.modules())
        assert MODEL_INFO["quantized_linear_count"] > 0
        MODEL_INFO["model_revision"] = provenance["revision"]
        MODEL_INFO["model_available"] = True
    except Exception as error:
        MODEL_INFO["load_error"] = f"{type(error).__name__}: {error}"
        MODEL_INFO["model_available"] = False
        MODEL = None
        gc.collect()
    MODEL_INFO.update(load_time_s=perf_counter() - start, process_rss_before_load_bytes=before,
                      process_rss_after_load_bytes=process.memory_info().rss,
                      ram_available_after_load_bytes=psutil.virtual_memory().available)
    return MODEL_INFO


def load_model():
    global MODEL, SERVER, SERVER_URL, SERVER_LOG
    start = perf_counter()
    folder = ROOT / "models/qwen2.5-0.5b-instruct-gguf"
    MODEL_INFO["backend"] = "llama.cpp CPU local, API limitée à 127.0.0.1"
    MODEL_INFO["quantization"] = "GGUF Q4_K_M officiel Qwen (quantification mixte 4 bits)"
    MODEL_INFO["parameters_approximate"] = 494_000_000
    MODEL_INFO["parameter_count_source"] = "Architecture officielle Qwen 0.5B ; fichier GGUF du fournisseur"
    MODEL_INFO["soft_timeout_s"] = TIMEOUT_S
    MODEL_INFO["initial_int8_attempt"] = "diagnostics/transformers_int8_first_attempt"
    try:
        if psutil.virtual_memory().available < 1_000_000_000:
            raise MemoryError("Moins de 1 Go libre : ne pas charger le modèle GGUF.")
        provenance = json.loads((folder / "download_provenance.json").read_text(encoding="utf-8"))
        for relative, entry in provenance["files"].items():
            assert rules.file_hash(ROOT / relative) == entry["sha256"]
        executable = ROOT / provenance["llama_server_path"]
        model_path = folder / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        with socket.socket() as channel:
            channel.bind(("127.0.0.1", 0))
            port = channel.getsockname()[1]
        SERVER_URL = f"http://127.0.0.1:{port}"
        log_folder = ROOT / "results/llm_agent"
        log_folder.mkdir(parents=True, exist_ok=True)
        SERVER_LOG = (log_folder / "llama_server.log").open("w", encoding="utf-8")
        SERVER = subprocess.Popen([str(executable), "-m", str(model_path), "--host", "127.0.0.1",
                                   "--port", str(port), "-t", "4", "-c", "4096", "-ngl", "0",
                                   "--parallel", "1"], cwd=executable.parent,
                                  stdout=SERVER_LOG, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while perf_counter() - start < 60:
            if SERVER.poll() is not None:
                raise RuntimeError("llama-server arrêté : consulter llama_server.log.")
            try:
                with opener.open(SERVER_URL + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.25)
        else:
            raise TimeoutError("Chargement du serveur local trop long.")
        MODEL = "llama.cpp_server_ready"
        MODEL_INFO.update(model_available=True, model_revision=provenance["revision"],
                          gguf_model_id=provenance["model_id"], llama_cpp_release=provenance["llama_cpp_release"],
                          server_rss_after_load_bytes=psutil.Process(SERVER.pid).memory_info().rss,
                          model_file_bytes=model_path.stat().st_size, json_schema_constrained_generation=True)
    except Exception as error:
        MODEL_INFO.update(model_available=False, load_error=f"{type(error).__name__}: {error}")
        stop_model()
    MODEL_INFO.update(load_time_s=perf_counter() - start,
                      ram_available_after_load_bytes=psutil.virtual_memory().available)
    return MODEL_INFO


def stop_model():
    global SERVER, SERVER_LOG, MODEL
    if SERVER is not None and SERVER.poll() is None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=5)
        except subprocess.TimeoutExpired:
            SERVER.kill()
            SERVER.wait(timeout=5)
    if SERVER_LOG is not None:
        SERVER_LOG.close()
    SERVER = None
    SERVER_LOG = None
    MODEL = None


import atexit
atexit.register(stop_model)


def extract_json(text):
    return guardrails.extract_json(text)


def _unique_pairs(pairs):
    return guardrails._unique_pairs(pairs)


def validate_decision(output):
    # Préserver le contrat historique du backend : confidence doit être un float.
    return guardrails.validate_llm_decision(output, validator=rules.validate_decision)


def qualitative_checks(output, state):
    return guardrails.qualitative_checks(output, state, rules=rules, validator=validate_decision)


def llm_decision(state):
    global LAST_CALL
    LAST_CALL = {"raw_text": None, "parsed_output": None, "error": None,
                 "inference_time_s": None, "input_tokens": None, "output_tokens": None}
    start = perf_counter()
    try:
        if MODEL is None:
            raise RuntimeError("Modèle local indisponible : " + str(MODEL_INFO.get("load_error")))
        baseline = rules.rule_based_decision(state)
        payload = {"state": state, "indicators": rules.indicators(state),
                   "traffic_state_allowed": sorted(rules.TRAFFIC_STATES),
                   "risk_level_allowed": sorted(rules.RISK_LEVELS),
                   "allowed_actions_here": list(policy.LLM_GENERATION_ALLOWED_ACTIONS),
                   "minimum_risk": baseline["risk_level"],
                   "current_occupancy_six_decimals": f"{state['current_occupancy']:.6f}",
                   "predicted_occupancy_six_decimals": f"{state['predicted_occupancy_t15']:.6f}"}
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]
        schema = {"type": "object", "properties": {
            "traffic_state": {"type": "string", "enum": sorted(rules.TRAFFIC_STATES)},
            "risk_level": {"type": "string", "enum": sorted(rules.RISK_LEVELS)},
            "recommended_action": {"type": "string", "enum": payload["allowed_actions_here"]},
            "reason": {"type": "string", "maxLength": policy.LLM_REASON_MAX_LENGTH},
            "confidence": {"type": "number", "minimum": policy.CONFIDENCE_MINIMUM,
                           "maximum": policy.CONFIDENCE_MAXIMUM}},
            "required": sorted(rules.DECISION_KEYS), "additionalProperties": False}
        request_payload = {"messages": messages, "temperature": 0, "max_tokens": 300,
                           "response_format": {"type": "json_schema", "json_schema": {
                               "name": "goflow_decision", "strict": True, "schema": schema}}}
        request = urllib.request.Request(SERVER_URL + "/v1/chat/completions",
                                         data=json.dumps(request_payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=TIMEOUT_S) as response:
            answer = json.load(response)
        LAST_CALL["input_tokens"] = answer.get("usage", {}).get("prompt_tokens")
        LAST_CALL["output_tokens"] = answer.get("usage", {}).get("completion_tokens")
        LAST_CALL["finish_reason"] = answer["choices"][0].get("finish_reason")
        text = answer["choices"][0]["message"]["content"]
        LAST_CALL["raw_text"] = text
        result = extract_json(text)
        LAST_CALL["parsed_output"] = result
        if perf_counter() - start > TIMEOUT_S:
            raise TimeoutError("Délai souple d'inférence dépassé ; sortie conservée pour diagnostic.")
        return result
    except Exception as error:
        LAST_CALL["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        LAST_CALL["inference_time_s"] = perf_counter() - start


def final_decision(state):
    return decision_engine.final_saved_scenario_decision(
        state, inputs=INPUTS, llm_backend=llm_decision,
        qualitative_checker=qualitative_checks, rule_decision=rules.rule_based_decision)


def run_scenarios():
    records, comparisons, validations, times = [], [], [], []
    for item in INPUTS:
        state = item["state"]
        result = final_decision(state)
        call = copy.deepcopy(LAST_CALL)
        baseline = rules.rule_based_decision(state)
        record = {"scenario": item["scenario"], "state": state, "rule_based_output": baseline,
                  **result, "llm_call": call}
        records.append(record)
        llm_action = (result["llm_output"] or {}).get("recommended_action")
        comparisons.append({"Scenario": item["scenario"], "Rule-based action": baseline["recommended_action"],
                            "LLM action": llm_action, "Same action?": llm_action == baseline["recommended_action"],
                            "LLM JSON valid?": result["validation"]["json_valid"],
                            "Hallucination detected?": result["validation"]["hallucination_detected"],
                            "Reason grounded?": result["validation"]["reason_grounded"],
                            "Inference time": call["inference_time_s"], "Final source": result["source"]})
        validations.append({"scenario": item["scenario"], **result["validation"],
                            "final_source": result["source"], "fallback_reason": result["fallback_reason"]})
        times.append({"scenario": item["scenario"], "inference_time_s": call["inference_time_s"],
                      "input_tokens": call["input_tokens"], "output_tokens": call["output_tokens"],
                      "error": call["error"]})
        print(item["scenario"], "source=", result["source"], "temps=", call["inference_time_s"], flush=True)
    return records, pd.DataFrame(comparisons), pd.DataFrame(validations), pd.DataFrame(times)


def save_results(records, comparisons, validations, times, methodology):
    folder = ROOT / "results/llm_agent"
    folder.mkdir(parents=True, exist_ok=True)
    MODEL_INFO["process_rss_after_inference_bytes"] = psutil.Process().memory_info().rss
    if SERVER is not None and SERVER.poll() is None:
        MODEL_INFO["server_rss_after_inference_bytes"] = psutil.Process(SERVER.pid).memory_info().rss
    # Fermer explicitement le serveur avant de terminer le notebook, pas seulement à la sortie du noyau.
    stop_model()
    MODEL_INFO["server_stopped_after_inference"] = True
    MODEL_INFO["technical_integration_success"] = bool(
        MODEL_INFO.get("model_available") and all(row["llm_call"]["raw_text"] is not None for row in records))
    for filename, data in [("llm_model_info.json", MODEL_INFO), ("llm_scenario_outputs.json", records)]:
        (folder / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    comparisons.to_csv(folder / "llm_vs_rules.csv", index=False)
    validations.to_csv(folder / "llm_validation_results.csv", index=False)
    times.to_csv(folder / "llm_inference_times.csv", index=False)
    (folder / "llm_agent_methodology.md").write_text(methodology, encoding="utf-8")
    summary = {"Modèle LLM utilisé": MODEL_INFO["model_id"], "Device": MODEL_INFO["device"],
               "Nombre de scénarios": len(records), "JSON valides": int(validations.json_valid.sum()),
               "Fallbacks": int((validations.final_source == "rule_based_fallback").sum()),
               "Hallucinations détectées sur JSON analysables": int(validations.hallucination_detected.sum()),
               "Réponses non analysables": int((~validations.hallucination_check_applicable).sum()),
               "Temps moyen d'inférence (s)": float(times.inference_time_s.mean()),
               "Intégration technique": MODEL_INFO["technical_integration_success"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
