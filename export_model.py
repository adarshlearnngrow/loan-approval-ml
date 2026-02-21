"""
One-time script: exports the registered MLflow model to models/model.pkl
and saves its metadata (threshold, run_id) to models/model_info.json.
Run once from the project root:
    python export_model.py
"""
import json
import shutil
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

SRC_MODEL = (
    PROJECT_ROOT
    / "mlartifacts"
    / "286339840280967947"
    / "models"
    / "m-d2140a0a189542b6a976da0d0ffe27d6"
    / "artifacts"
    / "model.pkl"
)

DEST_DIR = PROJECT_ROOT / "models"
DEST_DIR.mkdir(exist_ok=True)

DEST_MODEL = DEST_DIR / "model.pkl"
DEST_INFO  = DEST_DIR / "model_info.json"

# ── Copy model.pkl ─────────────────────────────────────────────────────────────
shutil.copy2(SRC_MODEL, DEST_MODEL)
print(f"Copied model.pkl  ({DEST_MODEL.stat().st_size / 1024:.1f} KB)")

# ── Read f1_threshold from mlruns ──────────────────────────────────────────────
thresh_file = (
    PROJECT_ROOT
    / "mlruns"
    / "286339840280967947"
    / "203160c728f14070a42b7b8b28a72905"
    / "metrics"
    / "f1_threshold"
)
threshold = 0.5
if thresh_file.exists():
    parts = thresh_file.read_text().strip().split()
    threshold = float(parts[1]) if len(parts) >= 2 else float(parts[0])
    print(f"f1_threshold      = {threshold}")
else:
    print("f1_threshold not found, defaulting to 0.5")

# ── Write model_info.json ──────────────────────────────────────────────────────
info = {
    "run_id":        "203160c728f14070a42b7b8b28a72905",
    "model_name":    "Credit Risk Model Final",
    "model_version": "2",
    "f1_threshold":  threshold,
}
DEST_INFO.write_text(json.dumps(info, indent=2))
print(f"Written model_info.json: {info}")
print("\nDone. Commit the models/ folder to git.")
