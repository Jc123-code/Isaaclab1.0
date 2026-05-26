#!/usr/bin/env python3

"""Merge demo groups from two HDF5 files and rename them as demo_0, demo_1, ...

Expected default structure:

    data/
      demo_0/
      demo_1/

The demo root can also be "/" when demo groups are stored at the file root.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py


DEMO_RE = re.compile(r"^demo_(\d+)$")


def demo_sort_key(name: str) -> int:
    """Return numeric suffix from demo_N."""
    match = DEMO_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid demo name: {name}")
    return int(match.group(1))


def normalize_root(root: str) -> str:
    """Normalize CLI root names for h5py."""
    if root in ("", "/"):
        return "/"
    return root.strip("/")


def get_demo_names(h5_file: h5py.File, demo_root: str) -> list[str]:
    """Get demo_N group names sorted by N."""
    root_group = h5_file if demo_root == "/" else h5_file[demo_root]
    demo_names = [name for name in root_group.keys() if DEMO_RE.match(name)]
    return sorted(demo_names, key=demo_sort_key)


def copy_attrs(src: h5py.Group | h5py.File, dst: h5py.Group | h5py.File) -> None:
    """Copy HDF5 attributes from one object to another."""
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def merge_hdf5_files(input_files: list[Path], output_file: Path, demo_root: str, overwrite: bool) -> None:
    """Merge demo groups from input files into output_file."""
    demo_root = normalize_root(demo_root)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --overwrite to replace it.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as fout:
        output_root = fout if demo_root == "/" else fout.create_group(demo_root)

        demo_counter = 0
        copied_root_attrs = False

        for input_file in input_files:
            with h5py.File(input_file, "r") as fin:
                if demo_root != "/" and demo_root not in fin:
                    raise KeyError(f"{input_file} does not contain group: {demo_root}")

                input_root = fin if demo_root == "/" else fin[demo_root]

                if not copied_root_attrs:
                    copy_attrs(input_root, output_root)
                    copied_root_attrs = True

                demo_names = get_demo_names(fin, demo_root)
                if not demo_names:
                    print(f"[WARN] No demo_N groups found in {input_file} under {demo_root}")
                    continue

                for old_demo_name in demo_names:
                    new_demo_name = f"demo_{demo_counter}"
                    fin.copy(input_root[old_demo_name], output_root, name=new_demo_name)
                    print(f"{input_file}: {old_demo_name} -> {new_demo_name}")
                    demo_counter += 1

        output_root.attrs["num_demos"] = demo_counter

    print(f"Done. Merged {demo_counter} demos into: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge demo_N groups from HDF5 files and reorder them as demo_0, demo_1, ...",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input HDF5 files. Example: a.hdf5 b.hdf5",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--demo-root",
        default="data",
        help='Group containing demo_N groups. Default: "data". Use "/" if demos are at file root.',
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Please provide at least two input HDF5 files.")

    for input_file in args.inputs:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        if input_file.resolve() == args.output.resolve():
            raise ValueError(
                f"Output file must be different from every input file to avoid overwriting data: {args.output}"
            )

    merge_hdf5_files(args.inputs, args.output, args.demo_root, args.overwrite)


if __name__ == "__main__":
    main()
