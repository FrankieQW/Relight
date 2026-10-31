from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .config import ProjectConfig
from .io_utils import (
    dump_json_atomic,
    dump_jsonl_atomic,
    load_json,
    sha256_file,
    utc_now,
)
from .scene_semantics import normalize_label
from .schemas import ObjectDocument


_GLB_PATH = re.compile(
    r"^(?P<shard>[0-9]{3}-[0-9]{3})/(?P<uid>[0-9a-fA-F]{32})\.glb$"
)
_LICENSE_NAMES = {
    "by": "CC-BY-4.0",
    "by-nc": "CC-BY-NC-4.0",
    "by-nc-sa": "CC-BY-NC-SA-4.0",
    "by-sa": "CC-BY-SA-4.0",
    "cc0": "CC0-1.0",
}


def prepare_objects(config: ProjectConfig) -> dict[str, Any]:
    inventory_path = config.path("object_inventory")
    lvis_path = config.path("lvis_annotations")
    object_paths_path = config.path("object_paths")
    metadata_path = config.path("object_metadata")
    output_path = config.path("object_output")
    rejects_path = config.path("object_rejects")

    inventory, inventory_rejects = _read_inventory(inventory_path)
    lvis = load_json(lvis_path)
    object_paths = load_json(object_paths_path)
    metadata = load_json(metadata_path)
    if not isinstance(lvis, dict) or not isinstance(object_paths, dict) or not isinstance(metadata, dict):
        raise TypeError("LVIS annotations, object paths and metadata must all be JSON objects")

    uid_categories: dict[str, list[str]] = defaultdict(list)
    for category, uids in lvis.items():
        if not isinstance(uids, list):
            raise TypeError(f"LVIS category '{category}' must map to a list")
        for uid in uids:
            normalized_uid = str(uid).lower()
            uid_categories[normalized_uid].append(str(category))

    selected: list[dict[str, Any]] = []
    rejects = list(inventory_rejects)
    for uid, inventory_relative_path in sorted(inventory.items()):
        categories = sorted(set(uid_categories.get(uid, [])))
        if not categories:
            rejects.append(
                {
                    "uid": uid,
                    "inventory_path": inventory_relative_path,
                    "reason": "not_in_lvis_annotations",
                }
            )
            continue

        canonical_path = object_paths.get(uid)
        if not canonical_path:
            rejects.append(
                {
                    "uid": uid,
                    "inventory_path": inventory_relative_path,
                    "reason": "missing_object_path",
                }
            )
            continue

        primary_category = categories[0]
        metadata_row = metadata.get(uid)
        selected.append(
            _build_object_row(
                uid=uid,
                inventory_path=inventory_relative_path,
                canonical_path=str(canonical_path).replace("\\", "/"),
                categories=categories,
                metadata=metadata_row if isinstance(metadata_row, dict) else None,
            )
        )

    unique_categories = sorted({row["primary_category_normalized"] for row in selected})
    object_config = config.section("object")
    expected_count = int(object_config.get("expected_selected_count", len(selected)))
    expected_categories = int(
        object_config.get("expected_unique_category_count", len(unique_categories))
    )
    if bool(object_config.get("strict_expected_counts", True)):
        if len(selected) != expected_count:
            raise ValueError(
                f"Expected {expected_count} selected objects, got {len(selected)}. "
                "The strict intersection may have changed."
            )
        if len(unique_categories) != expected_categories:
            raise ValueError(
                f"Expected {expected_categories} normalized categories, got {len(unique_categories)}"
            )

    license_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        license_counts[row["license"] or "missing"] += 1

    document = {
        "schema_version": str(config.section("project").get("schema_version", "1.0")),
        "generated_at": utc_now(),
        "generator_version": __version__,
        "config_digest": config.digest,
        "inventory_mode": "markdown_lvis_strict_intersection",
        "object_root_env": "OBJECT_ROOT",
        "source_digests": {
            "inventory": sha256_file(inventory_path),
            "lvis_annotations": sha256_file(lvis_path),
            "object_paths": sha256_file(object_paths_path),
            "metadata": sha256_file(metadata_path),
        },
        "stats": {
            "inventory_glbs": len(inventory),
            "annotated_unique_uids": len(uid_categories),
            "selected_objects": len(selected),
            "unique_selected_categories": len(unique_categories),
            "rejected_objects": len(rejects),
            "license_counts": dict(sorted(license_counts.items())),
        },
        "objects": selected,
    }
    document = ObjectDocument.model_validate(document).model_dump(mode="json")
    dump_json_atomic(output_path, document)
    dump_jsonl_atomic(rejects_path, rejects)
    return document


def _read_inventory(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    inventory: dict[str, str] = {}
    rejects: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        value = raw_line.strip().replace("\\", "/")
        if not value.lower().endswith(".glb"):
            continue
        if value.startswith("/") or ".." in Path(value).parts:
            rejects.append(
                {"line": line_number, "value": value, "reason": "unsafe_inventory_path"}
            )
            continue
        match = _GLB_PATH.fullmatch(value)
        if match is None:
            rejects.append(
                {"line": line_number, "value": value, "reason": "invalid_inventory_path"}
            )
            continue
        uid = match.group("uid").lower()
        if uid in inventory:
            rejects.append(
                {
                    "line": line_number,
                    "uid": uid,
                    "value": value,
                    "reason": "duplicate_inventory_uid",
                }
            )
            continue
        inventory[uid] = f"{match.group('shard')}/{uid}.glb"
    return inventory, rejects


def _build_object_row(
    *,
    uid: str,
    inventory_path: str,
    canonical_path: str,
    categories: list[str],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = metadata or {}
    primary_category = categories[0]
    license_raw = metadata.get("license")
    user = metadata.get("user") if isinstance(metadata.get("user"), dict) else {}
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
    tag_names = sorted(
        {
            str(tag.get("name"))
            for tag in tags
            if isinstance(tag, dict) and tag.get("name")
        }
    )
    thumbnails = metadata.get("thumbnails") if isinstance(metadata.get("thumbnails"), dict) else {}
    images = thumbnails.get("images") if isinstance(thumbnails.get("images"), list) else []
    thumbnail_rows = sorted(
        (
            {
                "url": image.get("url"),
                "width": image.get("width"),
                "height": image.get("height"),
            }
            for image in images
            if isinstance(image, dict) and image.get("url")
        ),
        key=lambda row: int(row.get("width") or 0),
        reverse=True,
    )[:3]
    archives = metadata.get("archives") if isinstance(metadata.get("archives"), dict) else {}
    glb_archive = archives.get("glb") if isinstance(archives.get("glb"), dict) else None

    author = {
        key: user[key]
        for key in ("uid", "username", "displayName", "profileUrl")
        if key in user
    }
    return {
        "uid": uid,
        "primary_category": primary_category,
        "primary_category_normalized": normalize_label(primary_category),
        "categories": categories,
        "categories_normalized": [normalize_label(category) for category in categories],
        "inventory_path": inventory_path,
        "canonical_path": canonical_path,
        "shard": inventory_path.split("/", 1)[0],
        "license_raw": license_raw,
        "license": _LICENSE_NAMES.get(str(license_raw), str(license_raw) if license_raw else None),
        "source_uri": metadata.get("uri"),
        "embed_url": metadata.get("embedUrl"),
        "name": metadata.get("name"),
        "description": metadata.get("description"),
        "tags": tag_names,
        "author": author,
        "is_age_restricted": metadata.get("isAgeRestricted"),
        "is_downloadable": metadata.get("isDownloadable"),
        "glb_stats": glb_archive,
        "thumbnails": thumbnail_rows,
        "metadata_status": "available" if metadata else "missing",
    }
