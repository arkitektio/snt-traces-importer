"""SNT ``.traces`` files: what Fiji's Simple Neurite Tracer saves, read without a server.

A ``.traces`` file is XML, usually gzipped: one ``<samplespacing>`` and one ``<imagesize>``
for the image it was traced on, then one ``<path>`` per traced neurite. A path's
``<point x y z>`` are **pixel indices** of that image (``xd yd zd`` are the same points
times the spacing), in SNT's ``x, y, z`` order. ``channel`` and ``frame`` are 1-based.

A path may branch off another (``startson`` + ``startsindex``: the parent and the vertex of
it the branch leaves from), and SNT may keep a smoothed copy of a path beside it: the copy
says ``fittedversionof="<original id>"``.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class SntPath:
    """One traced path, as SNT saved it."""

    id: int
    name: str
    #: ``(n, 3)`` pixel indices in SNT's ``x, y, z`` order.
    points: np.ndarray
    #: 1-based, as SNT counts.
    channel: int
    #: 1-based, as SNT counts.
    frame: int
    swc_type: int
    color: str | None
    real_length: float | None
    starts_on: int | None
    starts_index: int | None
    ends_on: int | None
    ends_index: int | None
    #: Set on SNT's smoothed copy of a path: the id of the path it smooths.
    fitted_version_of: int | None


@dataclass(frozen=True)
class Tracings:
    """A whole ``.traces`` file."""

    #: ``(x, y, z)`` voxel size in ``units``.
    spacing: tuple[float, float, float]
    units: str
    #: ``(width, height, depth)`` of the image the paths were traced on.
    image_size: tuple[int, int, int]
    paths: tuple[SntPath, ...]


def _int(value: str | None) -> int | None:
    if value is None or value.strip() in ("", "-1", "None"):
        return None
    return int(float(value))


def _float(value: str | None) -> float | None:
    if value is None or value.strip() in ("", "None", "NaN"):
        return None
    return float(value)


def _color(value: str | None) -> str | None:
    # SNT writes the literal string "None" for a path it never coloured.
    if value is None or value.strip() in ("", "None"):
        return None
    return value.strip()


def _read_bytes(path: str | Path) -> bytes:
    raw = Path(path).read_bytes()
    return gzip.decompress(raw) if raw[:2] == GZIP_MAGIC else raw


def read_tracings(path: str | Path) -> Tracings:
    """Read a ``.traces`` file, gzipped or plain.

    Raises:
        ValueError: If the file is not an SNT tracing (no ``<tracings>`` root, or no
            ``<imagesize>`` to check it against an image).
    """
    root = ET.fromstring(_read_bytes(path))
    if root.tag != "tracings":
        raise ValueError(f"{Path(path).name} is not an SNT tracing: its root is <{root.tag}>, not <tracings>.")

    size = root.find("imagesize")
    if size is None:
        raise ValueError(f"{Path(path).name} names no <imagesize>, so it cannot be checked against an image.")
    image_size = (int(size.get("width")), int(size.get("height")), int(size.get("depth")))

    spacing_element = root.find("samplespacing")
    if spacing_element is not None:
        spacing = tuple(float(spacing_element.get(axis, "1")) for axis in "xyz")
        units = spacing_element.get("units", "pixel")
    else:
        spacing, units = (1.0, 1.0, 1.0), "pixel"

    paths = []
    for element in root.findall("path"):
        points = np.array(
            [[float(point.get(axis)) for axis in "xyz"] for point in element.findall("point")],
            dtype=np.float64,
        ).reshape(-1, 3)
        path_id = int(element.get("id"))
        paths.append(
            SntPath(
                id=path_id,
                name=element.get("name") or f"Path ({path_id})",
                points=points,
                channel=_int(element.get("channel")) or 1,
                frame=_int(element.get("frame")) or 1,
                swc_type=_int(element.get("swctype")) or 0,
                color=_color(element.get("color")),
                real_length=_float(element.get("reallength")),
                starts_on=_int(element.get("startson")),
                starts_index=_int(element.get("startsindex")),
                ends_on=_int(element.get("endson")),
                ends_index=_int(element.get("endsindex")),
                fitted_version_of=_int(element.get("fittedversionof")),
            )
        )

    return Tracings(spacing=spacing, units=units, image_size=image_size, paths=tuple(paths))
