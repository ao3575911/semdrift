"""Causal LM loading: pull once, then run offline from the semdrift model cache."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semdrift.corpus.workspace import cache_root
from semdrift.errors import ModelError, ResourceError, missing_extra

DEFAULT_MODEL = "distilbert/distilgpt2"
# Licences that allow research use and redistribution of outputs without extra terms.
OPEN_LICENCES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0", "openrail"}


def models_dir() -> Path:
    return cache_root() / "models"


def local_path(name: str) -> Path:
    p = Path(name).expanduser()
    if p.is_dir():
        return p
    return models_dir() / name.replace("/", "--")


def _require() -> None:
    for mod in ("torch", "transformers"):
        try:
            __import__(mod)
        except ImportError as e:
            raise missing_extra(mod, "lm") from e


def card_licence(repo: str) -> str:
    from huggingface_hub import model_info

    try:
        info = model_info(repo)
    except Exception as e:  # network, auth, 404: all surface the same way to the user
        raise ResourceError(
            f"could not look up '{repo}' on the Hugging Face Hub ({type(e).__name__})",
            "check the model id (e.g. distilbert/distilgpt2) and your network",
        ) from e
    card = info.card_data
    lic = card.get("license") if card is not None else None
    return str(lic or "unknown").lower()


def pull(repo: str, accept_licence: bool = False) -> tuple[Path, str]:
    """Download a model snapshot into the cache after checking its licence."""
    _require()
    from huggingface_hub import snapshot_download

    lic = card_licence(repo)
    if lic not in OPEN_LICENCES and not accept_licence:
        raise ModelError(
            f"'{repo}' is licensed '{lic}', which is not on the open list",
            "read the licence, then rerun with --accept-licence",
        )
    dest = local_path(repo)
    try:
        snapshot_download(
            repo,
            local_dir=dest,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt"],
        )
    except Exception as e:
        raise ResourceError(
            f"download of '{repo}' failed: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        ) from e
    if not any(dest.glob("*.safetensors")):
        raise ModelError(f"'{repo}' has no safetensors weights", "pick a model that ships them")
    (dest / "semdrift.json").write_text(json.dumps({"repo": repo, "licence": lic}))
    return dest, lic


@dataclass
class LM:
    name: str
    model: Any
    tokenizer: Any
    max_len: int
    bos: int


def load(name: str) -> LM:
    _require()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]
    hf_logging.disable_progress_bar()  # type: ignore[no-untyped-call]

    path = local_path(name)
    if not (path / "config.json").exists():
        raise ModelError(
            f"model '{name}' is not available offline",
            f"run: semdrift lm pull {name}",
        )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    try:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    except Exception as e:
        raise ModelError(
            f"could not load '{name}': {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        ) from e
    if not getattr(tok, "is_fast", False):
        raise ModelError(f"'{name}' has no fast tokenizer", "word alignment needs offsets")
    model.train(False)
    max_len = int(getattr(model.config, "n_positions", 0) or model.config.max_position_embeddings)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if bos is None:
        raise ModelError(f"'{name}' has no BOS/EOS token to start a context")
    return LM(name, model, tok, min(max_len, 1024), int(bos))
