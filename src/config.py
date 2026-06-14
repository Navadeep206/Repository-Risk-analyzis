import os

# Prevent OpenMP runtime conflict crashes on macOS (Apple Silicon / Intel)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Root directories
BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR: str = os.path.join(BASE_DIR, "data")

RAW_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")
REPOS_DIR: str = os.path.join(DATA_DIR, "repositories")
FINAL_DIR: str = os.path.join(DATA_DIR, "final")

def ensure_dirs_exist() -> None:
    """
    Ensures that all the required data directories exist.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPOS_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

