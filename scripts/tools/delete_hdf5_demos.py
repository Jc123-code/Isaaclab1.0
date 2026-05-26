#!/usr/bin/env python3

"""Delete selected demonstration groups from an Isaac Lab HDF5 dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py


DEMO_RE = re.compile(r"^demo_(\d+)$")


def _parse_demo_token(token: str) -> list[str]:
    """Parse demo tokens such as ``3``, ``demo_3``, ``3,5`` or ``3-5``."""
    demo_keys: list[str] = []
    for part in token.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            start_idx = _parse_demo_index(start)
            end_idx = _parse_demo_index(end)
            if end_idx < start_idx:
                raise ValueError(f"Invalid descending demo range: {part}")
            demo_keys.extend(f"demo_{idx}" for idx in range(start_idx, end_idx + 1))
        else:
            demo_keys.append(f"demo_{_parse_demo_index(part)}")
    return demo_keys


def _parse_demo_index(text: str) -> int:
    text = text.strip()
    if text.startswith("demo_"):
        text = text.removeprefix("demo_")
    return int(text)


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = DEMO_RE.match(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def _selected_demo_keys(tokens: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        for demo_key in _parse_demo_token(token):
            if demo_key not in seen:
                selected.append(demo_key)
                seen.add(demo_key)
    return selected


def _update_total_attr(data_group: h5py.Group) -> None:
    total = 0
    for episode_name in data_group.keys():
        episode = data_group[episode_name]
        total += int(episode.attrs.get("num_samples", 0))
    data_group.attrs["total"] = total


def _delete_mask_entries(h5_file: h5py.File, deleted_keys: set[str]) -> None:
    if "mask" not in h5_file:
        return

    for mask_name, dataset in list(h5_file["mask"].items()):
        kept = [key for key in dataset.asstr()[()] if key not in deleted_keys]
        del h5_file["mask"][mask_name]
        encoded = [key.encode("utf-8") for key in kept]
        h5_file["mask"].create_dataset(mask_name, data=encoded)


def _renumber_demos(data_group: h5py.Group) -> dict[str, str]:
    """Renumber demo groups to demo_0...demo_N and return old-to-new mapping."""
    old_names = sorted(data_group.keys(), key=_demo_sort_key)
    mapping = {old_name: f"demo_{idx}" for idx, old_name in enumerate(old_names)}

    # Move to temporary names first so existing target names cannot collide.
    temp_mapping = {}
    for old_name in old_names:
        temp_name = f"__tmp_delete_hdf5_demos_{old_name}"
        data_group.move(old_name, temp_name)
        temp_mapping[temp_name] = mapping[old_name]

    for temp_name, new_name in temp_mapping.items():
        data_group.move(temp_name, new_name)

    return mapping


def _renumber_mask_entries(h5_file: h5py.File, mapping: dict[str, str]) -> None:
    if "mask" not in h5_file:
        return

    for mask_name, dataset in list(h5_file["mask"].items()):
        renamed = [mapping[key] for key in dataset.asstr()[()] if key in mapping]
        del h5_file["mask"][mask_name]
        encoded = [key.encode("utf-8") for key in renamed]
        h5_file["mask"].create_dataset(mask_name, data=encoded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete selected demo groups, for example /data/demo_3, from an HDF5 dataset."
    )
    parser.add_argument("--dataset_file", required=True, type=Path, help="Path to the .hdf5 dataset.")
    parser.add_argument(
        "--demos",
        required=True,
        nargs="+",
        help="Demo ids to delete. Examples: 3 7 demo_12, or 3,7,12, or 3-8.",
    )
    parser.add_argument("--data_group", default="data", help="HDF5 group containing demos. Default: data.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify the file. Without this flag the script only prints what would be deleted.",
    )
    parser.add_argument(
        "--renumber",
        action="store_true",
        help="After deletion, rename remaining demos to demo_0...demo_N without gaps.",
    )
    args = parser.parse_args()

    selected = _selected_demo_keys(args.demos)
    mode = "r+" if args.apply else "r"

    with h5py.File(args.dataset_file, mode) as h5_file:
        if args.data_group not in h5_file:
            raise KeyError(f"Cannot find group '/{args.data_group}' in {args.dataset_file}")

        data_group = h5_file[args.data_group]
        existing = set(data_group.keys())
        found = [key for key in selected if key in existing]
        missing = [key for key in selected if key not in existing]

        print(f"Dataset: {args.dataset_file}")
        print(f"Found demos to delete: {found}")
        if missing:
            print(f"Missing demos skipped: {missing}")

        if not args.apply:
            print("Dry run only. Add --apply to modify the file.")
            return

        for key in found:
            del data_group[key]

        _delete_mask_entries(h5_file, set(found))

        if args.renumber:
            mapping = _renumber_demos(data_group)
            _renumber_mask_entries(h5_file, mapping)
            print(f"Renumbered {len(mapping)} remaining demos.")

        _update_total_attr(data_group)
        h5_file.flush()

        print(f"Deleted {len(found)} demos.")
        print(f"Remaining demos: {len(data_group.keys())}")
        print(f"/{args.data_group}.attrs['total'] = {data_group.attrs['total']}")


if __name__ == "__main__":
    main()
