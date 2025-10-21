# AC Advisor — Environment & Quickstart

This repo runs a **Streamlit-based Interactive What-If + AI Coach** for A/C energy prediction using your CatBoost model.

## Two ways to set up

### Option A — Conda/Mamba (recommended for local dev)

```bash
# If you have mamba:
mamba env create -f environment.yml
mamba activate ac-advisor

# Or with conda:
conda env create -f environment.yml
conda activate ac-advisor
```

### Option B — Docker (portable & deployable)

```bash
# Build
docker build -t ac-advisor .

# Run
docker run --rm -p 8501:8501 -e TZ=UTC ac-advisor
```

## Run the app

```bash
# From the project root
streamlit run app.py
```

## Project layout (suggested)

```
.
├── app.py                    # Streamlit UI (replay + what-if + coach)
├── models/
│   └── catboost_model.cbm    # Trained CatBoost model (exported from your script)
├── data/
│   └── EnergyPredictionDataset_ReadyForModel.csv
├── ac_advisor/
│   ├── __init__.py
│   ├── features.py           # make_features(row) — single source for features
│   ├── predictor.py          # loads model, predict_now(), predict_sim()
│   ├── comfort.py            # ΔT bands, comfort scoring
│   └── coach.py              # nearest-context recall + logging
├── logs/
│   └── interactions.jsonl    # (auto-created) user/coach action logs
├── environment.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Notes
- Python **3.11**.
- GPU **not required** (CatBoost CPU is fast for tabular prediction).
- Stick to **`features.py`** for a consistent feature recipe across training, replay, and simulation.
- Keep your trained CatBoost file at `models/catboost_model.cbm` (or change the path in `predictor.py`).