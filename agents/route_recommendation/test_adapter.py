from adapter_traffic import load_approach_map, forecast_to_alerts

amap = load_approach_map()
msg = {"type": "forecast", "t": 1530.0,
       "nodes": {"1427-1": {"fill": [0.6, 0.8, 0.9], "halt": [9, 12, 14]}}}
print(forecast_to_alerts(msg, amap))