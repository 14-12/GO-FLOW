"""Agent GoFlow hors ligne : prévisions sauvegardées, règles et interface LLM contrainte."""
# %% Imports et contrat
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import numbers
import shutil
import sys
import urllib.request

import numpy as np
import pandas as pd

from src.decision import decision_engine, guardrails, policy

# Préserver des ensembles et une configuration indépendants pour chaque runtime.
TRAFFIC_STATES = set(policy.TRAFFIC_STATES)
RISK_LEVELS = set(policy.RISK_LEVELS)
ACTIONS = set(policy.ACTIONS)
DECISION_KEYS = set(policy.DECISION_KEYS)
CONFIG = {**policy.default_rule_config(), "llm_timeout_s": 30,
          "llm_model": None, "max_local_model_bytes": 2_000_000_000,
          "minimum_available_ram_bytes": 2_500_000_000}
MAPPING_GROUPS = [[48, 61, 80, 121, 122, 123, 141, 142],
                  [50, 51, 74, 75, 90, 91, 115, 131],
                  [55, 70, 73, 81, 83, 84, 159], [62, 109]]
SYSTEM_PROMPT = """Tu es l'agent Traffic Prediction & Adaptive Signal Control de GoFlow.
Tu reçois des observations réelles, une prévision XGBoost à +15 minutes, des tendances
strictement passées, des voisins directs et des indicateurs expérimentaux.
Ne modifie aucune valeur. N'invente aucun incident, véhicule d'urgence, voisin ou donnée.
Les incidents, urgences et plans de feux sont inconnus : signale ces informations manquantes.
Ne propose aucune durée de feu. Ne prétends pas à une précision scientifique calibrée.
Produis uniquement un objet JSON avec exactement traffic_state, risk_level,
recommended_action, reason, confidence. Respecte le contrat fourni : état et risque imposés,
action parmi allowed_actions, reason copiée exactement parmi allowed_reasons,
confidence égale à la valeur imposée. Le choix d'une justification est contraint aux faits
calculés ; aucune justification libre n'est autorisée dans cette première version.
"""


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_environment(root):
    # Vérifier l'environnement sans installation et sans téléchargement.
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ["torch", "transformers", "llama_cpp", "ollama", "psutil"]}
    result = {"python": sys.version, "packages": packages,
              "ollama_executable": shutil.which("ollama"),
              "gguf_files": [str(p.relative_to(root)) for p in (root / "models").rglob("*.gguf")]}
    if packages["torch"]:
        import torch
        result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    if packages["psutil"]:
        import psutil
        memory = psutil.virtual_memory()
        result.update(ram_total_bytes=memory.total, ram_available_bytes=memory.available)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            result["ollama_models"] = json.load(response).get("models", [])
    except (OSError, ValueError) as error:
        result["ollama_models"] = []
        result["ollama_status"] = type(error).__name__
    # Un modèle instruct doit être explicitement identifié et assez petit pour cette RAM.
    eligible = [m for m in result["ollama_models"]
                if "instruct" in m.get("name", "").lower()
                and 0 < m.get("size", 0) <= CONFIG["max_local_model_bytes"]]
    if result.get("ram_available_bytes", 0) >= CONFIG["minimum_available_ram_bytes"] and eligible:
        CONFIG["llm_model"] = min(eligible, key=lambda m: m["size"])["name"]
    result["selected_llm"] = CONFIG["llm_model"]
    result["llm_mode"] = "ollama_local" if CONFIG["llm_model"] else "placeholder_rule_based"
    return result

# %% Chargement et validation des données
def initialize(root):
    global ROOT, PREDICTIONS, OBSERVATIONS, ADJACENCY, LOOKUP, THRESHOLDS, ENVIRONMENT
    ROOT = Path(root).resolve()
    ENVIRONMENT = inspect_environment(ROOT)
    folder = ROOT / "results/model_benchmark/xgboost"
    PREDICTIONS = pd.read_csv(folder / "xgboost_test_predictions.csv")
    required = {"timestep", "location", "current_flow", "current_speed", "current_occupancy",
                "target_occupy_t15", "predicted_occupancy_t15"}
    if not required.issubset(PREDICTIONS.columns):
        raise ValueError(f"Colonnes absentes : {required - set(PREDICTIONS.columns)}")
    assert len(PREDICTIONS) == policy.HISTORICAL_PREDICTION_COUNT
    assert np.isfinite(PREDICTIONS[list(required)].to_numpy()).all()
    expected = pd.MultiIndex.from_product([range(policy.SENSOR_COUNT), range(policy.HISTORICAL_FIRST_TIMESTEP, policy.HISTORICAL_LAST_TIMESTEP + 1)],
                                         names=["location", "timestep"]).to_frame(index=False)
    pd.testing.assert_frame_equal(PREDICTIONS[["location", "timestep"]], expected)
    metadata = json.loads((folder / "xgboost_export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["target"] == "target_occupy_t15" and metadata["horizon_minutes"] == policy.FORECAST_HORIZON_MINUTES
    assert metadata["n_test"] == len(PREDICTIONS)
    raw_path = ROOT / "data/raw/traffic_dataset.csv"
    assert file_hash(raw_path) == metadata["source_hashes"]["data\\raw\\traffic_dataset.csv"]
    raw = pd.read_csv(raw_path).sort_values(["location", "timestep"]).reset_index(drop=True)
    assert len(raw) == policy.SOURCE_TIMESTEP_COUNT * policy.SENSOR_COUNT and not raw.duplicated(["location", "timestep"]).any()
    assert np.array_equal(raw.location.to_numpy(), np.repeat(np.arange(policy.SENSOR_COUNT), policy.SOURCE_TIMESTEP_COUNT))
    assert np.array_equal(raw.timestep.to_numpy(), np.tile(np.arange(policy.SOURCE_TIMESTEP_ORIGIN, policy.SOURCE_TIMESTEP_COUNT + 1), policy.SENSOR_COUNT))
    assert np.isfinite(raw[["flow", "speed", "occupy"]].to_numpy()).all()
    # Stocker (capteurs, temps, features), sans faire de normalisation ni d'inférence.
    OBSERVATIONS = raw[["flow", "speed", "occupy"]].to_numpy().reshape(policy.SENSOR_COUNT, policy.SOURCE_TIMESTEP_COUNT, policy.TRAFFIC_FEATURE_COUNT)
    ids = PREDICTIONS.location.to_numpy(dtype=int)
    times = PREDICTIONS.timestep.to_numpy(dtype=int) - policy.SOURCE_TIMESTEP_ORIGIN
    for feature, column in enumerate(["current_flow", "current_speed", "current_occupancy"]):
        np.testing.assert_allclose(PREDICTIONS[column], OBSERVATIONS[ids, times, feature],
                                   rtol=policy.SOURCE_RELATIVE_TOLERANCE, atol=policy.SOURCE_ABSOLUTE_TOLERANCE)
    # La cible future sert seulement à cet audit d'alignement ; elle n'entre jamais dans state.
    np.testing.assert_allclose(PREDICTIONS.target_occupy_t15, OBSERVATIONS[ids, times + policy.FORECAST_HORIZON_STEPS, policy.OCCUPANCY_FEATURE_INDEX],
                               rtol=policy.SOURCE_RELATIVE_TOLERANCE, atol=policy.SOURCE_ABSOLUTE_TOLERANCE)
    del raw
    ADJACENCY = np.load(ROOT / "results/model_benchmark/stgcn_adjacency.npy", allow_pickle=False)
    assert ADJACENCY.shape == (policy.SENSOR_COUNT, policy.SENSOR_COUNT) and np.isfinite(ADJACENCY).all()
    assert (ADJACENCY >= policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD).all() and np.allclose(ADJACENCY, ADJACENCY.T)
    assert np.count_nonzero(np.triu(ADJACENCY, 1)) == policy.SPATIAL_EDGE_COUNT
    assert ((ADJACENCY > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD).sum(axis=1) > 0).all()
    # Comparer la connectivité aux arêtes officielles, indépendamment des poids très petits.
    distance_path = ROOT / "data/source/pems08/d5d40645fa45471f2b473fc49a29ec3f0a993c6d/distance.csv"
    distances = pd.read_csv(distance_path)
    edges = np.zeros((policy.SENSOR_COUNT, policy.SENSOR_COUNT), dtype=bool)
    edges[distances["from"].to_numpy(dtype=int), distances["to"].to_numpy(dtype=int)] = True
    assert np.array_equal(ADJACENCY > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD, edges | edges.T)
    LOOKUP = PREDICTIONS.set_index(["location", "timestep"])
    # Ajuster tous les seuils sur la seule période train purgée du benchmark existant.
    first, last = CONFIG["train_first_timestep"], CONFIG["train_last_timestep"]
    train = OBSERVATIONS[:, first - 1:last, :]
    q50, q90, q97 = np.quantile(train[:, :, policy.OCCUPANCY_FEATURE_INDEX], CONFIG["occupancy_quantiles"])
    historical_changes = np.abs(OBSERVATIONS[:, first + policy.FORECAST_HORIZON_STEPS - 1:last + policy.FORECAST_HORIZON_STEPS, policy.OCCUPANCY_FEATURE_INDEX] - train[:, :, policy.OCCUPANCY_FEATURE_INDEX])
    assert last + policy.FORECAST_HORIZON_STEPS < policy.VALIDATION_FIRST_TIMESTEP
    THRESHOLDS = {"occupancy_median": float(q50), "occupancy_high": float(q90),
                  "occupancy_severe": float(q97),
                  "speed_low": float(np.quantile(train[:, :, policy.SPEED_FEATURE_INDEX], CONFIG["low_speed_quantile"])),
                  "speed_median": float(np.median(train[:, :, policy.SPEED_FEATURE_INDEX])),
                  "strong_change": float(np.quantile(historical_changes, CONFIG["change_quantile"]))}
    print("Prévisions validées :", PREDICTIONS.shape, "; graphe :", ADJACENCY.shape)
    return PREDICTIONS.head()

# %% Construction causale de l'état
def initialize_live(root):
    """Charger les seuils train et le graphe existants, sans CSV de prévisions ni cibles."""
    global ROOT, THRESHOLDS, ADJACENCY
    ROOT = Path(root).resolve()
    path = ROOT / "results/decision_agent/decision_agent_methodology.md"
    record = json.loads(path.read_text(encoding="utf-8").split("```json", 1)[1].split("```", 1)[0])
    thresholds = record["thresholds_train_only"]
    expected = {"occupancy_median", "occupancy_high", "occupancy_severe",
                "speed_low", "speed_median", "strong_change"}
    if set(thresholds) != expected or not all(
            isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
            for value in thresholds.values()):
        raise ValueError("Seuils train sauvegardés absents ou invalides.")
    adjacency = np.load(ROOT / "results/model_benchmark/stgcn_adjacency.npy", allow_pickle=False)
    if (adjacency.shape != (policy.SENSOR_COUNT, policy.SENSOR_COUNT) or not np.isfinite(adjacency).all()
            or not (adjacency >= policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD).all() or not np.allclose(adjacency, adjacency.T)
            or np.count_nonzero(np.triu(adjacency, 1)) != policy.SPATIAL_EDGE_COUNT
            or not ((adjacency > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD).sum(axis=1) > 0).all()):
        raise ValueError("Graphe spatial sauvegardé incompatible avec le contrat PEMS08.")
    THRESHOLDS = dict(thresholds)
    ADJACENCY = adjacency
    return dict(THRESHOLDS)


def build_traffic_state(sensor_id, timestep):
    if isinstance(sensor_id, bool) or isinstance(timestep, bool):
        raise ValueError("Identifiants entiers attendus.")
    if not isinstance(sensor_id, numbers.Integral) or not isinstance(timestep, numbers.Integral):
        raise ValueError("Identifiants entiers attendus.")
    sensor_id, timestep = int(sensor_id), int(timestep)
    if (sensor_id, timestep) not in LOOKUP.index:
        raise KeyError("Observation absente de l'export test XGBoost.")
    row = LOOKUP.loc[(sensor_id, timestep)]
    length = CONFIG["trend_history"]
    # Pour le timestep t, les six observations sont exactement t-6 à t-1.
    history = OBSERVATIONS[sensor_id, timestep - 1 - length:timestep - 1]
    assert len(history) == length
    neighbors = []
    for neighbor in np.flatnonzero(ADJACENCY[sensor_id] > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD):
        if neighbor == sensor_id:
            continue
        neighbor_row = LOOKUP.loc[(int(neighbor), timestep)]
        neighbors.append({"sensor_id": int(neighbor), "timestep": timestep,
                          "current_speed": float(neighbor_row.current_speed),
                          "current_occupancy": float(neighbor_row.current_occupancy),
                          "predicted_occupancy_t15": float(neighbor_row.predicted_occupancy_t15)})
    state = {"sensor_id": sensor_id, "timestep": timestep,
             "current_flow": float(row.current_flow), "current_speed": float(row.current_speed),
             "current_occupancy": float(row.current_occupancy),
             "predicted_occupancy_t15": float(row.predicted_occupancy_t15),
             "occupancy_change": float(row.predicted_occupancy_t15 - row.current_occupancy),
             "recent_occupancy_trend": float((history[-1, policy.OCCUPANCY_FEATURE_INDEX]
                                               - history[0, policy.OCCUPANCY_FEATURE_INDEX]) / (length - 1)),
             "recent_speed_trend": float((history[-1, policy.SPEED_FEATURE_INDEX]
                                           - history[0, policy.SPEED_FEATURE_INDEX]) / (length - 1)),
             "neighbors": neighbors, "incident_detected": None,
             "emergency_vehicle_present": None}
    assert "target_occupy_t15" not in state
    return state


def indicators(state):
    return decision_engine.indicators(state, THRESHOLDS)

# %% Règles et validation stricte
def rule_based_decision(state):
    return decision_engine.rule_based_decision(state, indicator_fn=indicators)


def strict_json_loads(text):
    return guardrails.strict_json_loads(text)


def decision_field_checks(data):
    return guardrails.decision_field_checks(
        data, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS, actions=ACTIONS)


def validate_decision(output):
    return guardrails.validate_decision(
        output, traffic_states=TRAFFIC_STATES, risk_levels=RISK_LEVELS,
        actions=ACTIONS, decision_keys=DECISION_KEYS)


def decision_contract(state):
    return decision_engine.decision_contract(
        state, rule_decision_fn=rule_based_decision, indicator_fn=indicators)


def is_grounded_decision(output, state):
    return guardrails.is_grounded_decision(
        output, state, validator=validate_decision, contract_builder=decision_contract,
        parser=strict_json_loads)

# %% Interface LLM et fallback
def llm_decision(state):
    # Placeholder remplaçable si aucun petit modèle instruct local n'est identifié.
    if not CONFIG["llm_model"]:
        return None
    payload = {"model": CONFIG["llm_model"], "stream": False, "format": "json",
               "options": {"temperature": 0, "num_predict": 400, "num_ctx": 4096},
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": json.dumps(
                                {"state": state, "contract": decision_contract(state)},
                                ensure_ascii=False, allow_nan=False)}]}
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=CONFIG["llm_timeout_s"]) as response:
        return json.load(response)["message"]["content"]


def final_decision(state):
    # Garder les points d'injection du notebook et le contexte propre à ce runtime.
    return decision_engine.final_decision(
        state, build_traffic_state=build_traffic_state, llm_decision=llm_decision,
        is_grounded_decision=is_grounded_decision, rule_based_decision=rule_based_decision,
        strict_json_loads=strict_json_loads)

# %% Sélection des cinq scénarios réels
def select_scenarios():
    table = PREDICTIONS.copy()
    table["occupancy_change"] = table.predicted_occupancy_t15 - table.current_occupancy
    # Quantifier le contraste courant entre voisins directs sans regarder les cibles futures.
    occupancy = table.current_occupancy.to_numpy().reshape(policy.SENSOR_COUNT, policy.HISTORICAL_TIMESTEP_COUNT)
    spread = np.empty_like(occupancy)
    for sensor in range(policy.SENSOR_COUNT):
        neighbors = np.flatnonzero(ADJACENCY[sensor] > policy.DIRECT_NEIGHBOR_WEIGHT_THRESHOLD)
        neighbors = neighbors[neighbors != sensor]
        spread[sensor] = np.ptp(occupancy[neighbors], axis=0)
    table["neighbor_occupancy_range"] = spread.reshape(-1)
    free = ((table.current_occupancy <= THRESHOLDS["occupancy_median"])
            & (table.predicted_occupancy_t15 <= THRESHOLDS["occupancy_median"])
            & (table.current_speed >= THRESHOLDS["speed_median"])
            & (table.occupancy_change < THRESHOLDS["strong_change"]))
    definitions = [
        ("trafic_fluide", free, "current_occupancy", True),
        ("occupation_elevee", table.current_occupancy >= THRESHOLDS["occupancy_high"],
         "current_occupancy", False),
        ("forte_hausse_predite", table.occupancy_change >= THRESHOLDS["strong_change"],
         "occupancy_change", False),
        ("amelioration_prevue", table.occupancy_change <= -THRESHOLDS["strong_change"],
         "occupancy_change", True),
        ("voisins_contrastes", table.neighbor_occupancy_range > 0,
         "neighbor_occupancy_range", False)]
    chosen, used = [], set()
    for name, mask, criterion, ascending in definitions:
        candidates = table.loc[mask & ~table.index.isin(used)].sort_values(
            criterion, ascending=ascending, kind="stable")
        if candidates.empty:
            raise ValueError(f"Aucun scénario réel satisfaisant : {name}")
        row = candidates.iloc[0].copy()
        used.add(candidates.index[0])
        row["scenario"] = name
        row["selection_criterion"] = criterion
        chosen.append(row)
    result = pd.DataFrame(chosen).reset_index(drop=True)
    for column in ["location", "timestep", "day_index", "slot_in_day", "hour", "minute"]:
        if column in result:
            result[column] = result[column].astype("int64")
    return result

# %% Validation des scénarios et sauvegarde
def evaluate_scenarios(scenarios):
    inputs, outputs, validations = [], [], []
    for _, row in scenarios.iterrows():
        state = build_traffic_state(int(row.location), int(row.timestep))
        result = final_decision(state)
        decision = result["decision"]
        inputs.append({"scenario": row.scenario, "state": state})
        outputs.append({"scenario": row.scenario, **result})
        flags = indicators(state)
        validations.append({"scenario": row.scenario, "source": result["source"],
                            "json_valid": validate_decision(json.dumps(decision, allow_nan=False)),
                            "authorized_action": decision["recommended_action"] in ACTIONS,
                            "canonical_input_verified": state == build_traffic_state(int(row.location), int(row.timestep)),
                            "reason_grounded_in_input": decision["reason"] in decision_contract(state)["allowed_reasons"],
                            "missing_information_signaled": "Incidents et urgences inconnus; plan de feux indisponible." in decision["reason"],
                            "risk_action_coherent": is_grounded_decision(decision, state),
                            "no_fabricated_data_in_contract": is_grounded_decision(decision, state),
                            "high_occupancy_neighbors": flags["congested_neighbors"]})
    return inputs, outputs, pd.DataFrame(validations)


def save_results(scenarios, inputs, outputs, validations, methodology):
    folder = ROOT / "results/decision_agent"
    folder.mkdir(parents=True, exist_ok=True)
    # Ne sauvegarder qu'après validation de toutes les sorties et sans écrire dans le benchmark.
    checks = ["json_valid", "authorized_action", "canonical_input_verified", "reason_grounded_in_input",
              "missing_information_signaled", "risk_action_coherent", "no_fabricated_data_in_contract"]
    assert validations[checks].to_numpy().all()
    scenarios.to_csv(folder / "selected_scenarios.csv", index=False)
    for filename, values in [("agent_inputs.json", inputs), ("agent_outputs.json", outputs)]:
        (folder / filename).write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False),
                                      encoding="utf-8")
    validations.to_csv(folder / "validation_results.csv", index=False)
    record = {"config": CONFIG, "thresholds_train_only": THRESHOLDS, "environment": ENVIRONMENT,
              "input_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                               [ROOT / "data/raw/traffic_dataset.csv",
                                ROOT / "results/model_benchmark/xgboost/xgboost_test_predictions.csv",
                                ROOT / "results/model_benchmark/stgcn_adjacency.npy"]},
              "decision_sources": validations.source.value_counts().to_dict()}
    (folder / "decision_agent_methodology.md").write_text(
        methodology + "\n\n## Configuration et exécution réelles\n\n```json\n"
        + json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n", encoding="utf-8")
    print("Résultats validés et sauvegardés :", folder)



def create_runtime():
    """Créer un contexte indépendant, sans chargement de données ni appel au LLM.

    Chaque agent possède ses propres données, seuils et configuration. Appeler
    initialize(root) explicitement pour vérifier et charger les exports existants.
    """
    spec = importlib.util.spec_from_file_location("_goflow_decision_runtime", __file__)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime
