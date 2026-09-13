#!/usr/bin/env python3
"""Inspect the structure and metadata of an HDF5 file.

The script recursively lists groups and datasets.  For datasets it reports the
shape, data type, size, storage layout, compression, and attributes without
loading the dataset values into memory.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-09-13
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import h5py
except ImportError:
    h5py = None


def format_value(value: Any, max_length: int = 120) -> str:
    """Return a compact printable representation of an attribute value.

    Parameters
    ----------
    value : Any
        Attribute value supplied by h5py.
    max_length : int, default=120
        Maximum number of characters retained in the representation.

    Returns
    -------
    str
        A one-line representation, shortened with an ellipsis if necessary.
    """
    text = repr(value).replace("\n", " ")
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def print_attributes(obj: h5py.Group, indent: str = "  ") -> None:
    """Print all attributes attached to an HDF5 group or dataset.

    Parameters
    ----------
    obj : h5py.Group or h5py.Dataset
        Open HDF5 object whose attributes are to be displayed.
    indent : str, default="  "
        Prefix inserted before every attribute line.

    Returns
    -------
    None
        The formatted attributes are written to standard output.
    """
    for key, value in obj.attrs.items():
        print(f"{indent}@{key}: {format_value(value)}")


def inspect_hdf5(filename: str) -> None:
    """Print a recursive summary of an HDF5 file without reading its data.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file to inspect.

    Returns
    -------
    None
        File, group, dataset, and attribute information is written to standard
        output.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    """
    if h5py is None:
        raise RuntimeError(
            "Missing dependency 'h5py'. Install it with "
            "'python -m pip install h5py' or 'conda install h5py'."
        )

    path = Path(filename).expanduser()

    with h5py.File(path, "r") as hdf:
        print(f"File: {path.resolve()}")
        print(f"Mode: {hdf.mode}")
        print("Root group: /")
        print_attributes(hdf)

        def show_item(name: str, obj: h5py.Group) -> None:
            """Print one object visited during recursive HDF5 traversal.

            Parameters
            ----------
            name : str
                Object path relative to the root group.
            obj : h5py.Group or h5py.Dataset
                HDF5 object found at ``name``.

            Returns
            -------
            None
                Information about the visited object is written to standard
                output.
            """
            depth = name.count("/")
            prefix = "  " * depth

            if isinstance(obj, h5py.Dataset):
                compression = obj.compression or "none"
                chunks = obj.chunks if obj.chunks is not None else "contiguous"
                print(
                    f"{prefix}[dataset] /{name}: shape={obj.shape}, "
                    f"dtype={obj.dtype}, elements={obj.size}, "
                    f"chunks={chunks}, compression={compression}"
                )
            else:
                print(f"{prefix}[group]   /{name} ({len(obj)} members)")

            print_attributes(obj, indent=prefix + "  ")

        hdf.visititems(show_item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse command-line arguments and run the HDF5 inspection.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments excluding the program name.  If ``None``, the
        arguments are read from ``sys.argv``.

    Returns
    -------
    int
        Zero after successful inspection.  Argument and file errors are
        reported by ``argparse`` and result in a nonzero process status.
    """
    parser = argparse.ArgumentParser(
        description="Recursively show groups, datasets, dimensions, and metadata."
    )
    parser.add_argument("filename", help="HDF5 file to inspect")
    args = parser.parse_args(argv)

    try:
        inspect_hdf5(args.filename)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
