"""Deterministic incident-to-traffic-prediction orchestrator."""
from pathlib import Path

from agents.incident_detection.src import IncidentDetector, process_video
from src.agents import TrafficPredictionControlAgent


def run_traffic_pipeline(video_path, *, source_id=None, observation_provider=None,
                         history_provider=None, incident_detector=None, prediction_agent=None,
                         video_processor=process_video, project_root=None):
    """Run Agent 1 first, then Agent 2 when genuine PEMS08 measurements exist.

    observation_provider(source_id, incident_result) must return sensor_id,
    timestep, and traffic_data with current_flow/current_speed/current_occupancy.
    history_provider(sensor_id, timestep) must return the causal PEMS08 history
    required by Agent 2 for the sensor and its graph neighbors.
    """
    source_id = source_id or Path(video_path).stem
    detector = incident_detector or IncidentDetector()
    incidents = detector.predict_windows(video_processor(video_path), source_id)
    agent = prediction_agent
    if agent is None and observation_provider is not None and history_provider is not None:
        agent = TrafficPredictionControlAgent(
            project_root=project_root, prediction_mode="live", history_provider=history_provider)
    output = []
    for incident in incidents:
        item = {"incident": incident, "handoff": None, "prediction": None}
        if observation_provider is None or history_provider is None:
            item["prediction_error"] = (
                "Live PEMS08 prediction requires an observation_provider with sensor_id, "
                "five-minute timestep, measured flow/speed/occupancy, and a causal "
                "history_provider for that sensor and its graph neighbors. Video motion "
                "features cannot supply those measurements.")
            output.append(item)
            continue
        try:
            observation = observation_provider(source_id, incident)
            if not isinstance(observation, dict):
                raise ValueError("observation_provider must return a dictionary")
            required = {"sensor_id", "timestep", "traffic_data"}
            if not required.issubset(observation):
                raise ValueError(f"Missing measured prediction fields: {sorted(required - set(observation))}")
            traffic = observation["traffic_data"]
            if not isinstance(traffic, dict):
                raise ValueError("traffic_data must be a dictionary of measured PEMS08 values")
            required_traffic = {"current_flow", "current_speed", "current_occupancy"}
            if not required_traffic.issubset(traffic):
                raise ValueError(f"Missing measured traffic fields: {sorted(required_traffic - set(traffic))}")
            message = {
                "request_id": f"{source_id}-{incident['window_start_sec']:g}",
                "sensor_id": observation["sensor_id"], "timestep": observation["timestep"],
                "traffic_data": {key: traffic[key] for key in sorted(required_traffic)},
                "incident_context": {"incident_detected": incident["incident_detected"],
                                     "incident_type": None, "severity": None},
            }
            if "timestamp" in observation:
                message["timestamp"] = observation["timestamp"]
            item["handoff"] = message
            prediction = agent.handle_message(message)
            item["prediction"] = prediction
            if prediction["status"] == "error":
                item["prediction_error"] = prediction["reason"]
        except Exception as error:
            item["prediction_error"] = f"{type(error).__name__}: {error}"
        output.append(item)
    return {"source_id": source_id, "windows": output}
