from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from src.chunking import ChunkRecord

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}  # in-process model cache, keyed by model_name


def load_embedding_model(model_name: str = "BAAI/bge-base-en-v1.5"):
    
    if model_name in _model_cache:
        return _model_cache[model_name]

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def _load_disk_cache(cache_path: Path, model_name: str) -> dict[str, np.ndarray]:
    
    if not cache_path.exists():
        return {}

    with open(cache_path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("model_name") != model_name:
        logger.warning(
            "Embedding cache at %s was built with model '%s', not '%s'. Ignoring stale cache.",
            cache_path, payload.get("model_name"), model_name,
        )
        return {}

    return payload.get("embeddings", {})


def _save_disk_cache(cache_path: Path, model_name: str, embeddings: dict[str, np.ndarray]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": model_name, "embeddings": embeddings}
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def embed_chunks(
    chunks: list[ChunkRecord],
    model_name: str = "BAAI/bge-base-en-v1.5",
    normalize: bool = True,
    batch_size: int = 32,
    cache_path: str | Path | None = "data/processed/embedding_cache.pkl",
) -> dict[str, np.ndarray]:
   
    cache_path = Path(cache_path) if cache_path is not None else None
    cached = _load_disk_cache(cache_path, model_name) if cache_path else {}

    to_compute = [c for c in chunks if c.chunk_id not in cached]
    logger.info(
        "embed_chunks: %d total chunks, %d already cached, %d to compute",
        len(chunks), len(chunks) - len(to_compute), len(to_compute),
    )

    if to_compute:
        model = load_embedding_model(model_name)
        texts = [c.text for c in to_compute]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
        )
        for chunk, vector in zip(to_compute, vectors):
            cached[chunk.chunk_id] = vector.astype(np.float32)

        if cache_path:
            _save_disk_cache(cache_path, model_name, cached)

    # Return only the embeddings relevant to this call's chunk set.
    result = {c.chunk_id: cached[c.chunk_id] for c in chunks}
    return result


def embeddings_to_matrix(
    chunks: list[ChunkRecord], embeddings: dict[str, np.ndarray]
) -> np.ndarray:
    
    return np.vstack([embeddings[c.chunk_id] for c in chunks])


def load_embedding_config(config_path: str = "configs/config.yaml") -> dict:
    
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["embedding"]