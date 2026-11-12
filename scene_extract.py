from __future__ import annotations

import hashlib
import html
import json
import logging
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .config import ProjectConfig
from .io_utils import (
    dump_json_atomic,
    dump_jsonl_atomic,
    load_json,
    relative_posix,
    sha256_file,
    stable_digest,
    utc_now,
)
from .scene_semantics import normalize_label
from .schemas import SceneDocument


LOGGER = logging.getLogger(__name__)


def prepare_scenes(
    config: ProjectConfig,
    *,
    blender_bin: str | None = None,
    workers: int | None = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    scene_config = config.section("scene")
    scene_root = config.path("scene_root")
    cache_root = config.path("scene_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    blender = str(blender_bin or scene_config.get("blender_bin"))
    if not blender:
        raise ValueError("scene.blender_bin is required")
    worker_count = max(1, int(workers or scene_config.get("workers", 1)))
    should_resume = bool(scene_config.get("resume", True) if resume is None else resume)
    _check_blender_version(blender, scene_config)

    aliases_path = _config_relative_path(config, str(scene_config["aliases_file"]))
    overrides_path = _config_relative_path(config, str(scene_config["overrides_file"]))
    aliases_document = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    aliases = aliases_document.get("aliases", {})
    overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
    if not isinstance(aliases, dict) or not isinstance(overrides, dict):
        raise TypeError("Scene aliases and overrides must be YAML mappings")

    fbx_files = sorted(scene_root.glob(str(scene_config.get("fbx_glob", "**/*.fbx"))))
    if not fbx_files:
        raise FileNotFoundError(f"No FBX files found under {scene_root}")

    # Resolve the Blender worker from this package, never from the config root:
    # the project directory may be renamed or live anywhere, and `config.root`
    # is free to point at an external data directory.
    script_path = Path(__file__).resolve().parent / "blender" / "entry.py"
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    jobs: list[dict[str, Any]] = []
    used_scene_ids: set[str] = set()
    for fbx_path in fbx_files:
        source_relative = fbx_path.resolve().relative_to(scene_root.resolve()).as_posix()
        scene_id = _unique_scene_id(_scene_id_from_path(fbx_path), used_scene_ids)
        used_scene_ids.add(scene_id)
        normalized_blend = cache_root / f"{scene_id}.blend"
        fragment_path = cache_root / "fragments" / f"{scene_id}.json"
        jobs.append(
            {
                "schema_version": str(config.section("project").get("schema_version", "1.0")),
                "generator_version": __version__,
                "config_digest": config.digest,
                "project_root": str(config.root),
                "source_fbx": str(fbx_path.resolve()),
                "source_relative": source_relative,
                "source_digest": sha256_file(fbx_path),
                "scene_id": scene_id,
                "normalized_blend": str(normalized_blend.resolve()),
                "normalized_blend_relative": relative_posix(normalized_blend, config.root),
                "fragment_path": str(fragment_path.resolve()),
                "id_hash_length": int(scene_config.get("id_hash_length", 16)),
                "aliases": aliases,
                "rename_overrides": overrides.get("rename_categories", {}),
                "support_categories": [
                    normalize_label(item)
                    for item in scene_config.get("support_categories", [])
                ],
                "non_replaceable_categories": [
                    normalize_label(item)
                    for item in scene_config.get("non_replaceable_categories", [])
                ],
                "ignore_source_lights": bool(scene_config.get("ignore_source_lights", False)),
            }
        )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _run_extract_job,
                job,
                blender,
                script_path,
                cache_root,
                should_resume,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - keep all scene failures in one report.
                LOGGER.exception("Scene extraction failed: %s", job["scene_id"])
                failures.append({"scene_id": job["scene_id"], "error": str(exc)})

    if failures:
        failure_path = cache_root / "scene_failures.json"
        dump_json_atomic(failure_path, {"generated_at": utc_now(), "failures": failures})
        raise RuntimeError(
            f"{len(failures)} scene(s) failed. See {failure_path} for details."
        )

    scenes = sorted((_apply_entity_overrides(row, overrides) for row in results), key=lambda x: x["scene_id"])
    total_nodes = sum(len(scene["nodes"]) for scene in scenes)
    total_entities = sum(len(scene["entities"]) for scene in scenes)
    output = {
        "schema_version": str(config.section("project").get("schema_version", "1.0")),
        "generated_at": utc_now(),
        "generator_version": __version__,
        "config_digest": config.digest,
        "source_digests": {
            scene["source_fbx"]: scene["source_digest"] for scene in scenes
        },
        "stats": {
            "scenes": len(scenes),
            "nodes": total_nodes,
            "entities": total_entities,
            "low_category_confidence": sum(
                entity["category_confidence"]
                < float(scene_config.get("low_confidence_threshold", 0.9))
                for scene in scenes
                for entity in scene["entities"]
            ),
        },
        "scenes": scenes,
    }
    _validate_scene_references(output)
    output = SceneDocument.model_validate(output).model_dump(mode="json")
    dump_json_atomic(config.path("scene_output"), output)
    review_rows = _build_review_rows(config, scenes)
    dump_jsonl_atomic(config.path("scene_review_jsonl"), review_rows)
    if bool(scene_config.get("render_review_images", False)):
        _render_review_assets(
            config=config,
            blender_bin=blender,
            script_path=script_path,
            scenes=scenes,
            review_rows=review_rows,
        )
    _write_review_html(config, review_rows)
    return output


def _run_extract_job(
    job: dict[str, Any],
    blender_bin: str,
    script_path: Path,
    cache_root: Path,
    resume: bool,
) -> dict[str, Any]:
    fragment_path = Path(job["fragment_path"])
    normalized_blend = Path(job["normalized_blend"])
    if resume and fragment_path.exists() and normalized_blend.exists():
        existing = load_json(fragment_path)
        if (
            existing.get("source_digest") == job["source_digest"]
            and existing.get("config_digest") == job["config_digest"]
        ):
            LOGGER.info("Reusing scene fragment for %s", job["scene_id"])
            return existing

    jobs_dir = cache_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_path = jobs_dir / f"{job['scene_id']}.json"
    dump_json_atomic(job_path, job)
    log_path = cache_root / "logs" / f"{job['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "extract",
        "--job",
        str(job_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(
            f"Blender exited with code {completed.returncode} for {job['scene_id']}:\n{tail}"
        )
    if not fragment_path.exists() or not normalized_blend.exists():
        raise RuntimeError(f"Blender did not produce expected outputs for {job['scene_id']}")
    return load_json(fragment_path)


def _apply_entity_overrides(scene: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    entities = {entity["entity_id"]: entity for entity in scene["entities"]}
    flags = overrides.get("entity_flags", {}) or {}
    for entity_id, patch in flags.items():
        if entity_id not in entities or not isinstance(patch, dict):
            continue
        for key in (
            "category",
            "category_confidence",
            "grouping_confidence",
            "replaceable",
            "support_surface",
        ):
            if key in patch:
                entities[entity_id][key] = patch[key]
        entities[entity_id]["override_applied"] = True

    for merge in overrides.get("merge_entities", []) or []:
        if not isinstance(merge, dict):
            continue
        member_ids = [item for item in merge.get("member_entity_ids", []) if item in entities]
        if len(member_ids) < 2:
            continue
        members = [entities.pop(item) for item in member_ids]
        entity_id = merge.get("entity_id") or _hash_id(
            f"{scene['scene_id']}\0merge\0{'|'.join(sorted(member_ids))}"
        )
        node_ids = sorted({node for member in members for node in member["node_ids"]})
        entities[entity_id] = {
            **members[0],
            "entity_id": entity_id,
            "node_ids": node_ids,
            "category": normalize_label(merge.get("category", members[0]["category"])),
            "grouping_confidence": 1.0,
            "grouping_method": "manual_merge_override",
            "obb_world": _bounds_for_nodes(scene, node_ids),
            "override_applied": True,
        }

    for split in overrides.get("split_entities", []) or []:
        if not isinstance(split, dict) or split.get("entity_id") not in entities:
            continue
        source = entities.pop(split["entity_id"])
        source_nodes = set(source["node_ids"])
        groups = split.get("groups", [])
        consumed: set[str] = set()
        created: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                continue
            node_ids = sorted(set(group.get("node_ids", [])) & source_nodes)
            if not node_ids or consumed.intersection(node_ids):
                continue
            consumed.update(node_ids)
            entity_id = group.get("entity_id") or (
                f"{scene['scene_id']}:entity:"
                + _hash_id(f"{source['entity_id']}\0split\0{index}\0{'|'.join(node_ids)}")
            )
            created.append(
                {
                    **source,
                    "entity_id": entity_id,
                    "node_ids": node_ids,
                    "category": normalize_label(group.get("category", source["category"])),
                    "grouping_confidence": 1.0,
                    "grouping_method": "manual_split_override",
                    "obb_world": _bounds_for_nodes(scene, node_ids),
                    "override_applied": True,
                }
            )
        if consumed != source_nodes:
            entities[source["entity_id"]] = source
            continue
        entities.update({entity["entity_id"]: entity for entity in created})

    scene["entities"] = sorted(entities.values(), key=lambda item: item["entity_id"])
    scene["stats"]["entities"] = len(scene["entities"])
    return scene


def _build_review_rows(config: ProjectConfig, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_config = config.section("scene")
    threshold = float(scene_config.get("low_confidence_threshold", 0.9))
    frequent = {normalize_label(value) for value in scene_config.get("frequent_review_categories", [])}
    frequent_minimum = int(scene_config.get("review_frequent_minimum", 20))
    fraction = float(scene_config.get("review_random_fraction", 0.1))
    minimum = int(scene_config.get("review_minimum_per_scene", 100))
    seed = int(config.section("project").get("seed", 0))
    rows: list[dict[str, Any]] = []

    for scene in scenes:
        entities = scene["entities"]
        reasons: dict[str, set[str]] = {entity["entity_id"]: set() for entity in entities}
        entity_by_id = {entity["entity_id"]: entity for entity in entities}
        for entity in entities:
            entity_id = entity["entity_id"]
            if entity["category_confidence"] < threshold:
                reasons[entity_id].add("low_category_confidence")
            if entity["grouping_confidence"] < threshold:
                reasons[entity_id].add("low_grouping_confidence")
            if len(entity["node_ids"]) > 1:
                reasons[entity_id].add("multi_node_entity")
            if entity.get("replaceable"):
                reasons[entity_id].add("replaceable")
            if entity.get("support_surface"):
                reasons[entity_id].add("support_surface")

        rng = random.Random(f"{seed}:{scene['scene_id']}")
        for category in frequent:
            candidates = [entity for entity in entities if entity["category"] == category]
            for entity in rng.sample(candidates, min(len(candidates), frequent_minimum)):
                reasons[entity["entity_id"]].add("frequent_category_sample")

        unselected = [entity for entity in entities if not reasons[entity["entity_id"]]]
        sample_size = min(len(unselected), max(round(len(entities) * fraction), minimum))
        for entity in rng.sample(unselected, sample_size):
            reasons[entity["entity_id"]].add("random_sample")

        for entity_id in sorted(reasons):
            if not reasons[entity_id]:
                continue
            entity = entity_by_id[entity_id]
            rows.append(
                {
                    "scene_id": scene["scene_id"],
                    "entity_id": entity_id,
                    "node_ids": entity["node_ids"],
                    "raw_label": entity["raw_label"],
                    "category": entity["category"],
                    "category_confidence": entity["category_confidence"],
                    "grouping_confidence": entity["grouping_confidence"],
                    "replaceable": entity["replaceable"],
                    "support_surface": entity["support_surface"],
                    "obb_dimensions": entity.get("obb_world", {}).get("dimensions"),
                    "review_reasons": sorted(reasons[entity_id]),
                    "allowed_verdicts": [
                        "pass",
                        "wrong_category",
                        "merge",
                        "split",
                        "not_replaceable",
                        "not_support_surface",
                    ],
                    "verdict": None,
                }
            )
    return rows


def _write_review_html(config: ProjectConfig, rows: list[dict[str, Any]]) -> None:
    output_path = config.path("scene_review_html")
    assets_root = config.path("scene_review_assets")
    table_rows: list[str] = []
    for row in rows:
        image_cells = []
        for view in ("context", "isolated", "top"):
            image_path = assets_root / row["scene_id"] / row["entity_id"] / f"{view}.png"
            if image_path.exists():
                relative = image_path.relative_to(output_path.parent).as_posix()
                image_cells.append(f'<img src="{html.escape(relative)}" width="220" loading="lazy">')
            else:
                image_cells.append("<span class=missing>未生成预览</span>")
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['scene_id'])}</td>"
            f"<td><code>{html.escape(row['entity_id'])}</code></td>"
            f"<td>{html.escape(row['raw_label'])}</td>"
            f"<td>{html.escape(row['category'])}</td>"
            f"<td>{row['category_confidence']:.2f} / {row['grouping_confidence']:.2f}</td>"
            f"<td>{html.escape(', '.join(row['review_reasons']))}</td>"
            f"<td>{''.join(image_cells)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Scene Entity 人工抽检</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:8px;vertical-align:top}}th{{position:sticky;top:0;background:#fff}}
code{{word-break:break-all}}.missing{{color:#888}}img{{margin:2px}}</style></head>
<body><h1>Scene Entity 人工抽检</h1>
<p>判定结果请写入 <code>configs/scene_overrides.yaml</code>，不要直接修改 scene.json。</p>
<p>共 {len(rows)} 个待审 entity。图片只有在启用可选的 review 渲染步骤后才会出现。</p>
<table><thead><tr><th>Scene</th><th>Entity ID</th><th>Raw label</th><th>Category</th>
<th>类别/分组置信度</th><th>抽检原因</th><th>Context / Isolated / Top</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def _render_review_assets(
    *,
    config: ProjectConfig,
    blender_bin: str,
    script_path: Path,
    scenes: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> None:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in review_rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    jobs_root = config.path("scene_cache") / "review_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    for scene_id, rows in sorted(by_scene.items()):
        scene = scene_by_id[scene_id]
        blend_path = config.root / scene["normalized_blend"]
        job_path = jobs_root / f"{scene_id}.json"
        job = {
            "scene_id": scene_id,
            "output_root": str(config.path("scene_review_assets")),
            "items": [
                {"entity_id": row["entity_id"], "node_ids": row["node_ids"]}
                for row in rows
            ],
        }
        dump_json_atomic(job_path, job)
        command = [
            blender_bin,
            "--background",
            str(blend_path),
            "--python",
            str(script_path),
            "--",
            "review",
            "--job",
            str(job_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_path = config.path("scene_cache") / "logs" / f"{scene_id}.review.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise RuntimeError(f"Review rendering failed for {scene_id}:\n{tail}")


def _bounds_for_nodes(scene: dict[str, Any], node_ids: list[str]) -> dict[str, Any]:
    node_index = {node["node_id"]: node for node in scene["nodes"]}
    bounds = [node_index[node_id]["aabb_world"] for node_id in node_ids if node_id in node_index]
    if not bounds:
        return {}
    minimum = [min(bound["min"][axis] for bound in bounds) for axis in range(3)]
    maximum = [max(bound["max"][axis] for bound in bounds) for axis in range(3)]
    center = [(minimum[axis] + maximum[axis]) / 2.0 for axis in range(3)]
    dimensions = [maximum[axis] - minimum[axis] for axis in range(3)]
    return {
        "method": "axis_aligned_union",
        "center": center,
        "dimensions": dimensions,
        "aabb": {"min": minimum, "max": maximum},
    }


def _validate_scene_references(document: dict[str, Any]) -> None:
    seen_scene_ids: set[str] = set()
    seen_entity_ids: set[str] = set()
    for scene in document.get("scenes", []):
        scene_id = scene["scene_id"]
        if scene_id in seen_scene_ids:
            raise ValueError(f"Duplicate scene_id: {scene_id}")
        seen_scene_ids.add(scene_id)
        node_ids = [node["node_id"] for node in scene.get("nodes", [])]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"Duplicate node_id in scene: {scene_id}")
        node_id_set = set(node_ids)
        referenced_nodes: set[str] = set()
        for entity in scene.get("entities", []):
            entity_id = entity["entity_id"]
            if entity_id in seen_entity_ids:
                raise ValueError(f"Duplicate entity_id: {entity_id}")
            seen_entity_ids.add(entity_id)
            missing = set(entity.get("node_ids", [])) - node_id_set
            if missing:
                raise ValueError(
                    f"Entity {entity_id} references missing node(s): {sorted(missing)}"
                )
            overlap = referenced_nodes.intersection(entity.get("node_ids", []))
            if overlap:
                raise ValueError(
                    f"Scene node(s) assigned to multiple entities in {scene_id}: {sorted(overlap)}"
                )
            referenced_nodes.update(entity.get("node_ids", []))
        if referenced_nodes != node_id_set:
            missing_entities = sorted(node_id_set - referenced_nodes)
            raise ValueError(
                f"Mesh node(s) have no entity in {scene_id}: {missing_entities[:20]}"
            )


def _config_relative_path(config: ProjectConfig, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config.root / path).resolve()


def _check_blender_version(blender_bin: str, scene_config: dict[str, Any]) -> None:
    completed = subprocess.run(
        [blender_bin, "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot execute Blender: {blender_bin}\n{completed.stdout}")
    match = re.search(r"Blender\s+(\d+\.\d+)", completed.stdout)
    if match is None:
        raise RuntimeError(f"Cannot parse Blender version from:\n{completed.stdout}")
    expected = str(scene_config.get("expected_blender_major_minor", "4.5"))
    if match.group(1) != expected and not bool(
        scene_config.get("allow_blender_version_mismatch", False)
    ):
        raise RuntimeError(
            f"Blender {expected}.x is required, but {match.group(1)} was found at {blender_bin}. "
            "Use the server Blender 4.5 build or explicitly allow a mismatch for diagnostics."
        )


def _scene_id_from_path(path: Path) -> str:
    value = re_camel_to_snake(path.stem)
    return normalize_label(value)


def re_camel_to_snake(value: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(value):
        if index and char.isupper() and (value[index - 1].islower() or value[index - 1].isdigit()):
            chars.append("_")
        chars.append(char)
    return "".join(chars)


def _unique_scene_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    return f"{candidate}_{index}"


def _hash_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]
