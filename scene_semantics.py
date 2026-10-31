from __future__ import annotations

import re
import unicodedata
from typing import Mapping


_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_TECHNICAL_TOKENS = {
    "a",
    "b",
    "exterior",
    "interior",
    "lod",
    "master",
    "mesh",
    "object",
    "paris",
    "research",
    "set",
}
_GENERIC_CATEGORY_TOKENS = {
    "antenna",
    "awning",
    "balcony",
    "basket",
    "bench",
    "bottle",
    "building",
    "cabinet",
    "ceiling",
    "chair",
    "counter",
    "curtain",
    "door",
    "floor",
    "flower",
    "glass",
    "lamp",
    "lantern",
    "light",
    "plate",
    "road",
    "shelf",
    "sign",
    "sidewalk",
    "stool",
    "table",
    "tree",
    "wall",
    "window",
}


def normalize_label(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value)
    value = _NON_ALNUM.sub("_", value.lower()).strip("_")
    return re.sub(r"_+", "_", value)


def classify_scene_name(
    raw_name: str,
    aliases: Mapping[str, str],
    rename_overrides: Mapping[str, str] | None = None,
) -> tuple[str, str, float]:
    normalized_name = normalize_label(raw_name)
    overrides = rename_overrides or {}
    if raw_name in overrides:
        return normalize_label(overrides[raw_name]), normalized_name, 1.0
    if normalized_name in overrides:
        return normalize_label(overrides[normalized_name]), normalized_name, 1.0

    stripped = re.sub(r"^bistro_research_(interior|exterior)_", "", normalized_name)
    stripped = _strip_instance_suffixes(stripped)

    normalized_aliases = {
        normalize_label(key): normalize_label(value) for key, value in aliases.items()
    }
    padded = f"_{stripped}_"
    for alias in sorted(normalized_aliases, key=len, reverse=True):
        if f"_{alias}_" in padded or stripped.startswith(f"{alias}_"):
            return normalized_aliases[alias], stripped, 0.99

    tokens = [
        token
        for token in stripped.split("_")
        if token
        and token not in _TECHNICAL_TOKENS
        and not _looks_like_instance_token(token)
    ]
    for token in tokens:
        if token in _GENERIC_CATEGORY_TOKENS:
            return token, stripped, 0.80

    fallback = "_".join(tokens[:4]) or "unknown"
    return fallback, stripped, 0.60 if fallback != "unknown" else 0.20


def category_has_affordance(category: str, configured: set[str]) -> bool:
    normalized = normalize_label(category)
    tokens = set(normalized.split("_"))
    return normalized in configured or bool(tokens & configured)


def _strip_instance_suffixes(value: str) -> str:
    previous = None
    while value != previous:
        previous = value
        value = re.sub(r"_(?:[0-9]+|[0-9a-f]{7,16})$", "", value)
        value = re.sub(r"_(?:mesh|lod[0-9]*|geo|geometry)$", "", value)
        value = value.rstrip("_")
    return value


def _looks_like_instance_token(token: str) -> bool:
    if token.isdigit():
        return True
    if re.fullmatch(r"[0-9a-f]{7,16}", token):
        return True
    if re.fullmatch(r"[a-z]?[0-9]+[a-z]?", token):
        return True
    return False

