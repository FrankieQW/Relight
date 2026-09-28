from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .config import load_config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .dataset import TokenLightKontextDataset


SPLITS = ("train", "validation", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize composed TokenLight samples before training runs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        action="append",
        help=(
            "train, validation or test; repeatable and comma separated; "
            "defaults to every split with an existing manifest"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes used for composing (default: 4)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report export completeness and exit non-zero when incomplete",
    )
    return parser.parse_args(argv)


def resolve_splits(values: list[str] | None) -> list[str]:
    if not values:
        return list(SPLITS)
    splits: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            if item not in SPLITS:
                raise ValueError(f"unknown split {item!r}; expected one of {SPLITS}")
            if item not in splits:
                splits.append(item)
    if not splits:
        raise ValueError("--split was given without a usable value")
    return splits


def select_splits(
    config: dict[str, object], requested: list[str]
) -> tuple[list[str], list[str]]:
    """Split the requested names into usable ones and ones without a manifest."""
    usable: list[str] = []
    skipped: list[str] = []
    paths = config["paths"]
    for split in requested:
        value = paths.get(f"{split}_manifest") if isinstance(paths, dict) else None
        if not value or not Path(str(value)).expanduser().is_file():
            skipped.append(split)
            continue
        usable.append(split)
    return usable, skipped


def count_complete(dataset: TokenLightKontextDataset) -> int:
    """Count samples whose export contains metadata plus both PNGs."""
    complete = 0
    for index in range(len(dataset)):
        sample_dir = dataset.sample_path(index)
        if sample_dir is not None and dataset.sample_is_complete(sample_dir):
            complete += 1
    return complete


def materialize_split(dataset: TokenLightKontextDataset, workers: int) -> None:
    """Drive __getitem__ so the export runs inside worker processes."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from .dataset import collate

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate,
        drop_last=False,
    )
    with tqdm(
        total=len(dataset), unit="sample", desc=f"compose {dataset.split}"
    ) as progress:
        for batch in loader:
            progress.update(int(batch["fixture_mask"].shape[0]))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    config = load_config(args.config)
    if not config["paths"].get("composed_output_root"):
        raise RuntimeError(
            "paths.composed_output_root must be configured to materialize composed samples"
        )

    requested = resolve_splits(args.split)
    splits, skipped = select_splits(config, requested)
    if not splits:
        raise RuntimeError(f"none of the requested splits has a manifest: {requested}")
    if skipped:
        print(
            json.dumps(
                {"splits_without_manifest": skipped},
                ensure_ascii=False,
            ),
            flush=True,
        )

    incomplete = False
    from .dataset import TokenLightKontextDataset

    for split in splits:
        dataset = TokenLightKontextDataset(config, split, materialize=True)
        expected = len(dataset)
        if not args.check:
            materialize_split(dataset, args.workers)
        complete = count_complete(dataset)
        print(
            json.dumps(
                {
                    "split": split,
                    "mode": "check" if args.check else "compose",
                    "export_id": dataset.export_id,
                    "composed_output_root": str(dataset.composed_output_root),
                    "expected": expected,
                    "complete": complete,
                    "missing": expected - complete,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if complete < expected:
            incomplete = True

    if incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
