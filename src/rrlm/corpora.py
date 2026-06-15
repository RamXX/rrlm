"""Natural-text corpora for tasks that synthetic generators cannot serve.

IMDB reviews (stanfordnlp/imdb) are fetched once through the HF datasets-server
REST API and cached locally. Real human text is the point: it cannot be
compressed to a template kernel, so per-item judgment is genuinely semantic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from rrlm.config import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
IMDB_CACHE = DATA_DIR / "imdb_pool.json"

_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_PAGE = 100  # API maximum page length
# train split layout of stanfordnlp/imdb: rows 0..12499 label=0, 12500..24999 label=1
_NEG_OFFSET, _POS_OFFSET = 0, 12_500


def _clean(text: str, max_chars: int = 500) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def fetch_imdb_pool(
    cache_path: Path = IMDB_CACHE, per_class: int = 2500, timeout: float = 30.0
) -> dict[str, list[str]]:
    """Return {'neg': [...], 'pos': [...]} review texts, fetching once and caching."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    pool: dict[str, list[str]] = {"neg": [], "pos": []}
    with httpx.Client(timeout=timeout) as client:
        for label, base_offset in (("neg", _NEG_OFFSET), ("pos", _POS_OFFSET)):
            for page_start in range(0, per_class, _PAGE):
                resp = client.get(
                    _ROWS_URL,
                    params={
                        "dataset": "stanfordnlp/imdb",
                        "config": "plain_text",
                        "split": "train",
                        "offset": base_offset + page_start,
                        "length": min(_PAGE, per_class - page_start),
                    },
                )
                resp.raise_for_status()
                for item in resp.json()["rows"]:
                    row = item["row"]
                    expected = 0 if label == "neg" else 1
                    if row["label"] != expected:
                        raise RuntimeError(
                            f"IMDB split layout assumption violated at offset "
                            f"{base_offset + page_start}: label {row['label']}"
                        )
                    pool[label].append(_clean(row["text"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pool))
    return pool
