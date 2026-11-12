"""Frozen T5 feature assets; no encoder or CUDA context is created in data workers."""

import functools
import hashlib
import json
import os
import pathlib

import numpy as np


def canonical_text(text: str) -> str:
    # Preserve case, punctuation, word order, and negation. Strip outer whitespace only.
    return text.strip()


@functools.lru_cache(maxsize=4)
def load_feature_cache(path: str) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        texts = archive["texts"].tolist()
        features = archive["features"].astype(np.float32)
        metadata = json.loads(str(archive["metadata"].item()))
    if features.ndim != 2 or len(texts) != len(features) or not np.isfinite(features).all():
        raise ValueError(f"Malformed semantic feature cache: {path}")
    if len(set(texts)) != len(texts):
        raise ValueError(f"Duplicate prompts in semantic cache: {path}")
    return dict(zip(texts, features, strict=True)), metadata


def file_sha256(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenT5Encoder:
    """Local-only encoder used for asset preparation and unseen inference prompts."""

    def __init__(self, model_dir: str, *, device: str = "cpu", max_length: int = 256):
        import torch
        from transformers import AutoTokenizer
        from transformers import T5EncoderModel

        self.torch = torch
        self.device = device
        self.max_length = max_length
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = T5EncoderModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.model.requires_grad_(requires_grad=False)

    def encode(self, prompts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokens.input_ids.shape[1] > self.max_length:
            raise ValueError("Instruction exceeds frozen text encoder length; refusing silent truncation")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return pooled.cpu().numpy().astype(np.float32)


@functools.lru_cache(maxsize=2)
def inference_encoder(model_dir: str, max_length: int) -> FrozenT5Encoder:
    return FrozenT5Encoder(model_dir, max_length=max_length)


def encode_inference_prompt(prompt: str, metadata: dict) -> np.ndarray:
    model_dir = os.environ.get("OPENPI_TEXT_MODEL_DIR", metadata["model_dir"])
    encoder = validated_inference_encoder(
        model_dir, metadata["max_length"], tuple(sorted(metadata["encoder_files_sha256"].items()))
    )
    return encoder.encode([prompt])[0]


@functools.lru_cache(maxsize=8192)
def prompt_feature(cache_path: str, prompt: str, feature_dim: int, *, allow_online: bool) -> np.ndarray:
    cache, metadata = load_feature_cache(cache_path)
    prompt = canonical_text(prompt)
    if prompt in cache:
        feature = cache[prompt]
    elif allow_online:
        feature = encode_inference_prompt(prompt, metadata)
    else:
        raise ValueError(f"Training prompt missing from frozen semantic cache: {prompt!r}")
    if feature.shape != (feature_dim,) or not np.isfinite(feature).all():
        raise ValueError("Invalid frozen semantic feature shape or values")
    return feature


@functools.lru_cache(maxsize=2)
def validated_inference_encoder(model_dir: str, max_length: int, hashes: tuple) -> FrozenT5Encoder:
    for name, expected in hashes:
        if file_sha256(pathlib.Path(model_dir) / name) != expected:
            raise ValueError(f"Frozen text encoder differs from training asset: {name}")
    return inference_encoder(model_dir, max_length)
