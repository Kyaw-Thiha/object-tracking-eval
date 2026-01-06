from pathlib import Path
import os

SRC_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", SRC_ROOT / "data"))
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", SRC_ROOT / "results"))
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", SRC_ROOT / "checkpoints"))
WORK_DIR_ROOT = Path(os.environ.get("WORK_DIR_ROOT", SRC_ROOT / "work_dirs"))
BDD_ROOT = Path(os.environ.get("BDD_ROOT", DATA_ROOT / "bdd100k"))
MOT17_ROOT = Path(os.environ.get("MOT17_ROOT", DATA_ROOT / "MOT17"))
CROWDHUMAN_ROOT = Path(os.environ.get("CROWDHUMAN_ROOT", DATA_ROOT / "crowdhuman"))
