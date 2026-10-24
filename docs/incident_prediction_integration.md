# Incident to prediction integration

The incident notebook was developed on AI City videos. Its reusable modules
are under `agents/incident_detection/src`. The prediction teammate's API is
`src.agents.TrafficPredictionControlAgent.handle_message(message)`. Its live
path calls `src.prediction.XGBoostTrafficPredictor.predict(history, sensor_id,
timestep)` and obtains sensor and direct-neighbor histories through a callable
`history_provider`. See `src/contracts/schemas.py` and
`src/prediction/xgboost_predictor.py` for their validated contracts.

From the repository root:

```python
from src.incident_pipeline import run_traffic_pipeline

result = run_traffic_pipeline(
    "highway.mp4", source_id="camera_01",
    observation_provider=my_measured_observation_provider,
    history_provider=my_pems08_history_provider,
)
```

The observation provider receives `(source_id, incident_result)` and returns
`{"sensor_id": 33, "timestep": 15227, "traffic_data":
{"current_flow": ..., "current_speed": ..., "current_occupancy": ...}}`.
These values must be measured values consistent with the history provider's
last row. The history provider receives `(sensor_id, timestep)` and returns a
chronological pandas DataFrame with `location`, `timestep`, `flow`, `speed`,
and `occupy`, at least from t-12 through t. The predictor validates sensor
identity, contiguous five-minute steps, causal ordering, feature schema, and
model metadata. Histories for graph neighbors are also needed.

The result contains ordered per-window incident evidence, the exact message
handed to Agent 2, and Agent 2's complete genuine response. Errors from Agent
2 are retained as a response and repeated in `prediction_error`. In the
absence of providers, the CLI still runs incident inference, sends a partial
message to the actual prediction agent, and retains its controlled validation
error with a null forecast. No LLM or emergency notification is
involved in the deterministic path.

The source branches do not contain a camera-to-PEMS08 sensor map or a live
feed of physical measurements. The detector runs at five-second resolution
with image-motion proxies, while the predictor is trained on five-minute
PEMS08 flow, speed, and occupancy. A real field integration must supply a
validated mapping and measured history; fabricating these values would make
the forecast scientifically invalid.
