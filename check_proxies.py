import json
from pathlib import Path

feature_path = Path("models/feature_columns.json")
if not feature_path.exists():
    print("❌ models/feature_columns.json not found — train/export your model first.")
    raise SystemExit

feature_cols = json.loads(feature_path.read_text())
print("\nTotal features in your model:", len(feature_cols))
proxies = ["temp_rpm_interaction", "speed", "humidity_over_voltage"]
print("\nChecking proxy columns:\n")
for p in proxies:
    print(f"{p:24} → {'✅ in model' if p in feature_cols else '❌ missing'}")
