# Incident Detection Agent

Traffic anomaly detection using YOLO11, ByteTrack, Random Forest, temporal features and LangChain.

## Pipeline

Video -> YOLO11 -> ByteTrack -> 5-second features -> Random Forest -> temporal confirmation -> LangChain agent.

## Model

The trained detector is stored in `models/incident_detector.joblib`.

## Evaluation

- Accuracy: 83.43%
- Precision: 24.30%
- Recall: 54.11%
- F1: 33.53%
- F2: 43.44%

This is an academic prototype and is not yet a production incident alert system.

## Integration

The notebook contains the current pipeline. Reusable inference functions and a structured handoff interface will be added for integration with the other GO-FLOW agents.
