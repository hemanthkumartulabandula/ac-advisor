import json, time
from pathlib import Path

LOG_PATH = Path("logs/interactions.jsonl")

def log_interaction(payload: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["t"] = int(time.time())
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")
