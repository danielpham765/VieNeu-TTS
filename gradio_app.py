import os
import sys

# Load environment variables from .env if present (for local dev convenience)
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")


def _ensure_writable_cache_env() -> None:
    cache_root = os.path.join(ROOT_DIR, ".cache")
    hf_root = os.path.join(cache_root, "huggingface")
    hub_root = os.path.join(hf_root, "hub")
    transformers_root = os.path.join(cache_root, "transformers")

    for path in (cache_root, hf_root, hub_root, transformers_root):
        os.makedirs(path, exist_ok=True)

    env_updates = {
        "XDG_CACHE_HOME": cache_root,
        "HF_HOME": hf_root,
        "HUGGINGFACE_HUB_CACHE": hub_root,
        "TRANSFORMERS_CACHE": transformers_root,
    }

    for key, target in env_updates.items():
        current = os.environ.get(key)
        if not current:
            os.environ[key] = target
            continue
        if not os.access(current, os.W_OK):
            os.environ[key] = target


_ensure_writable_cache_env()

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from apps.gradio_main import main


if __name__ == "__main__":
    main()
