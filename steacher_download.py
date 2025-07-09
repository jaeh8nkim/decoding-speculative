import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Configuration
USER = os.environ["USER"]
# SCRATCH = "scratch"
SCRATCH = "scratch2"
MODEL_ID = "simplescaling/s1.1-7B"

# Paths
BASE_DIR = Path(f"/{SCRATCH}/{USER}")
MODEL_DIR = BASE_DIR / "models" / "s1.1-7B"
CACHE_DIR = BASE_DIR / "hf_cache"

def main():
    # Set HF cache location
    os.environ["HF_HOME"] = str(CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {MODEL_ID} to {MODEL_DIR}")
    
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=str(CACHE_DIR),
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    
    print("Download complete!")

if __name__ == "__main__":
    main()