from typing import cast

import requests


def ollama_running(OLLAMA_URL: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=0.3)
        return cast(bool, r.status_code == 200)
    except requests.RequestException:
        return False
