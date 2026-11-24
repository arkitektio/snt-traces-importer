"""Imports SNT (Simple Neurite Tracer) ``.traces`` files as annotation collections.

One ``.traces`` file becomes one annotation collection over the image it was traced on:
one PATH per traced neurite (a POINT where SNT kept a single vertex), in that image's own
pixel grid. The collection is created the way ``ArrayLens.draw`` creates one: the image's
axes, an identity edge back to it, and the file as its source.

**The channel a path was traced on is not a pin.** A neurite is anatomy, not a property of
the marker it happened to be traced in, so it carries no ``c`` coordinate. Which channel
it was traced on is kept as ``tracedOn``: the lens of that channel, metadata only.

**A frame is a pin.** A path traced in frame 3 of a timelapse is where it is only at t = 2,
so it pins ``t`` (when the image has a ``t`` axis).

Everything is checked before anything is written. Nothing here is get-or-create, so a
failure half-way through would leave a half-filled collection.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from arkitekt import App, DescriptorOperator, Requires, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    AnnotationCollection,
    AnnotationKind,
    AnnotationRelation,
    AnnotationSourceInput,
    AnnotationSpecInput,
    ArrayDataset,
    AxisInput,
    CoordinateInput,
    CoordinateSystemDerivedFromInput,
    File,
    IdentityTransformInput,
    SourceFileInput,
)
from mikro.arkitekt.specs import EXTENSION

from snt_traces import SntPath, Tracings, read_tracings

logger = logging.getLogger(__name__)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.1.4"

app = App(
    "snt-traces-importer",
    __version__,
    description=(
        "Imports SNT (Simple Neurite Tracer) .traces files as annotation collections: one "
        "path per traced neurite, drawn in the image's own pixel grid, each remembering "
        "the channel it was traced on."
    ),
    author="jhnnsrs",
    services=[mikro_service],
)

#: How far a file's voxel size may stray from the image's before it is worth a warning.
SPACING_TOLERANCE = 0.01

#: The axes a vertex has a component along, in the order SNT does not use.
SPACE_AXES = ("z", "y", "x")

#: SNT's ``x, y, z`` column of each space axis.
SNT_COLUMN = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class PlannedShape:
    """One annotation about to be drawn, with nothing server-side resolved yet."""

    snt_id: int
    kind: AnnotationKind
    name: str
    description: str
    #: One row per vertex, one component per space axis of the image, in its axis order.
    vectors: list[list[float]]
    stroke_color: list[int] | None
    #: 0-based channel index, or None on an image without a ``c`` axis.
    channel: int | None
    #: 0-based timepoint the shape is pinned to, or None on an image without a ``t`` axis.
    t: int | None
    #: The SNT id of the path this one is SNT's smoothed copy of.
    refines: int | None


def _rgba(color: str | None) -> list[int] | None:
    """``#rrggbb`` (or ``#rrggbbaa``) as RGBA, or None for anything SNT may write instead."""
    if color is None or not re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", color):
        return None
    channels = [int(color[i : i + 2], 16) for i in range(1, len(color), 2)]
    return channels + [255] * (4 - len(channels))


def _description(path: SntPath, paths: dict[int, SntPath]) -> str:
    parts = [f"SNT path {path.id}", f"SWC type {path.swc_type}"]
    if path.real_length is not None:
        parts.append(f"{path.real_length:.3g} long as SNT measured it")
    if path.starts_on is not None:
        parent = paths.get(path.starts_on)
        where = (
            f" at its vertex {path.starts_index}"
            if path.starts_index is not None
            else ""
        )
        parts.append(
            f"branches off {parent.name if parent else f'path {path.starts_on}'}{where}"
        )
    if path.ends_on is not None:
        parent = paths.get(path.ends_on)
        where = (
            f" at its vertex {path.ends_index}" if path.ends_index is not None else ""
        )
        parts.append(
            f"ends on {parent.name if parent else f'path {path.ends_on}'}{where}"
        )
    if path.fitted_version_of is not None:
        parts.append(f"SNT's fitted version of path {path.fitted_version_of}")
    return " · ".join(parts)


def plan_annotations(
    tracings: Tracings,
    axis_names: Sequence[str],
    shape: Sequence[int],
    flip_z: bool,
) -> list[PlannedShape]:
    """Every path of `tracings` as a shape on an image of these axes and shape.

    Refuses, naming every problem at once, a tracing that does not fit the image: a
    different image size, a vertex outside it, a channel or frame it does not have.

    Raises:
        ValueError: If the tracing cannot have been made on this image.
    """
    sizes = dict(zip(axis_names, shape, strict=True))
    missing = [axis for axis in ("y", "x") if axis not in sizes]
    if missing:
        raise ValueError(
            f"The image has axes {list(axis_names)}: a tracing needs at least y and x, and {missing} are missing."
        )

    space = [axis for axis in axis_names if axis in SPACE_AXES]
    width, height, depth = tracings.image_size
    problems: list[str] = []

    expected = (sizes["x"], sizes["y"], sizes.get("z", 1))
    if (width, height, depth) != expected:
        problems.append(
            f"the tracing was made on an image of {width} x {height} x {depth} (x, y, z) and this one is "
            f"{expected[0]} x {expected[1]} x {expected[2]}: it was traced on another image"
        )

    paths = {path.id: path for path in tracings.paths}
    planned: list[PlannedShape] = []
    for path in tracings.paths:
        if len(path.points) == 0:
            continue

        points = path.points.copy()
        if flip_z and "z" in sizes:
            points[:, 2] = sizes["z"] - 1 - points[:, 2]

        # Voxel centres sit on the integers: an image spans [-0.5, size - 0.5] per axis.
        for axis in space:
            column = points[:, SNT_COLUMN[axis]]
            if column.min() < -0.5 or column.max() > sizes[axis] - 0.5:
                problems.append(
                    f"{path.name} leaves the image along {axis} ({column.min():g} to {column.max():g}, the image has {sizes[axis]})"
                )

        if "c" in sizes:
            if path.channel > sizes["c"]:
                problems.append(
                    f"{path.name} was traced on channel {path.channel} and the image has {sizes['c']}"
                )
        elif path.channel != 1:
            problems.append(
                f"{path.name} was traced on channel {path.channel} and the image has no channel axis"
            )

        if "t" in sizes:
            if path.frame > sizes["t"]:
                problems.append(
                    f"{path.name} was traced in frame {path.frame} and the image has {sizes['t']}"
                )
        elif path.frame != 1:
            problems.append(
                f"{path.name} was traced in frame {path.frame} and the image has no time axis"
            )

        refines = path.fitted_version_of if path.fitted_version_of in paths else None
        planned.append(
            PlannedShape(
                snt_id=path.id,
                kind=AnnotationKind.POINT if len(points) == 1 else AnnotationKind.PATH,
                name=path.name,
                description=_description(path, paths),
                # SNT's (x, y, z) columns, read out in the image's own axis order.
                vectors=points[:, [SNT_COLUMN[axis] for axis in space]].tolist(),
                stroke_color=_rgba(path.color),
                channel=path.channel - 1 if "c" in sizes else None,
                t=path.frame - 1 if "t" in sizes else None,
                refines=refines,
            )
        )

    if problems:
        raise ValueError(
            "The tracing does not fit this image: " + "; ".join(problems) + "."
        )
    return planned


def spacing_disagreement(tracings: Tracings, voxel_um: dict[str, float]) -> str | None:
    """A sentence saying how the file's voxel size differs from the image's, or None when they agree.

    Only a warning: Fiji may simply never have been told the calibration.
    """
    if tracings.units not in ("µm", "um", "micron", "microns", "micrometer"):
        return f"the tracing measures in {tracings.units!r}, so its spacing was not compared with the image's"
    file_um = dict(zip("xyz", tracings.spacing, strict=True))
    off = [
        f"{axis}: {file_um[axis]:g} µm in the file, {voxel_um[axis]:g} µm on the image"
        for axis in voxel_um
        if axis in file_um
        and voxel_um[axis]
        and abs(file_um[axis] - voxel_um[axis]) > SPACING_TOLERANCE * voxel_um[axis]
    ]
    return "; ".join(off) or None


def voxel_size_um(lens: object) -> dict[str, float]:
    """The lens' voxel size in µm per SPACE axis, as the server resolves it: ``{}`` when uncalibrated."""
    import pint

    registry = pint.UnitRegistry()
    sizes: dict[str, float] = {}
    for entry in getattr(lens, "voxel_size", None) or ():
        if str(getattr(entry.type, "value", entry.type)) == "SPACE":
            sizes[entry.axis] = float(
                (entry.size * registry(entry.unit)).to("micrometer").magnitude
            )
    return sizes


def _spec(
    shape: PlannedShape, traced_on: dict[int | None, str], originals: dict[int, str]
) -> AnnotationSpecInput:
    coordinates = (
        [CoordinateInput(name="t", value=shape.t)] if shape.t is not None else None
    )
    derived_from = None
    if shape.refines is not None:
        derived_from = [
            AnnotationSourceInput(
                annotation=originals[shape.refines],
                relation=AnnotationRelation.REFINED,
                reason="SNT's fitted version",
            )
        ]
    return AnnotationSpecInput(
        kind=shape.kind,
        name=shape.name,
        description=shape.description,
        vectors=shape.vectors,
        stroke_color=shape.stroke_color,
        coordinates=coordinates,
        derived_from=derived_from,
        traced_on=traced_on[shape.channel],
    )


@app.action(
    name="Import SNT Traces",
    description=(
        "Import an SNT (Simple Neurite Tracer) .traces file as an annotation collection on "
        "the image it was traced on: one path per traced neurite, in the image's pixel grid, "
        "each remembering the channel it was traced on."
    ),
    collections=["conversion"],
)
def import_snt_traces(
    file: Annotated[File, Requires(key=EXTENSION, operator=DescriptorOperator.IN, value=["traces"])],
    image: ArrayDataset,
    mikro: Mikro,
    task: Task,
    flip_z: bool,
) -> AnnotationCollection:
    """Import an SNT .traces file as an annotation collection.

    Args:
        file: The .traces file (gzipped or plain XML), as SNT saved it.
        image: The image the paths were traced on. Its x/y/z size must match the file's.
        flip_z: Turn the paths over in z. Set it when the image's z runs the other way
            from the file Fiji traced in: the ND2 converter turns a top-down stack over,
            and notes "z-order: reversed from top-down" on the image when it did.

    Returns:
        The new annotation collection.
    """
    task.progress(0, "Downloading the tracing")
    workdir = tempfile.mkdtemp(
        prefix="snt-traces-", dir=os.environ.get("MIKRO_SCRATCH_DIR") or None
    )
    try:
        path = file.store.download(os.path.join(workdir, os.path.basename(file.name)))
        tracings = read_tracings(path)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if flip_z and "z" not in image.axis_names:
        raise ValueError(
            f"flip_z asks to turn the paths over in z, and {image.name} has no z axis."
        )

    task.progress(20, f"Checking {len(tracings.paths)} path(s) against {image.name}")
    planned = plan_annotations(tracings, image.axis_names, image.shape, flip_z)
    if not planned:
        raise ValueError(f"{file.name} holds no path with a vertex.")

    whole = image.lens()
    disagreement = spacing_disagreement(tracings, voxel_size_um(whole))
    if disagreement:
        logger.warning(
            "%s and %s disagree about the voxel size (%s); importing by pixel index anyway",
            file.name,
            image.name,
            disagreement,
        )

    # One lens per channel a path was traced on, all before the first write: a channel the
    # server refuses then refuses the import, not its second half.
    traced_on: dict[int | None, str] = {None: whole.id}
    for channel in sorted({shape.channel for shape in planned} - {None}):
        traced_on[channel] = image.lens(c=channel).id

    task.progress(40, "Creating the collection")
    system = whole.coordinate_system
    stem = os.path.splitext(file.name)[0]
    collection = mikro.create_annotation_collection(
        name=f"{stem} · SNT tracings",
        description=(
            f"SNT tracings from {file.name}, drawn on {image.name}. Vertices are its level-0 pixel indices"
            + (", z turned over" if flip_z else "")
            + "."
        ),
        axes=[
            AxisInput(name=axis.name, type=axis.type, long_name=axis.long_name)
            for axis in system.axes
        ],
        derived_from=[
            CoordinateSystemDerivedFromInput(
                coordinate_system=system.id, transform=IdentityTransformInput()
            )
        ],
        source_files=[SourceFileInput(file=file.id)],
    )

    # The originals first: SNT's fitted copies name them as what they refine.
    task.progress(60, f"Drawing {len(planned)} shape(s)")
    originals: dict[int, str] = {}
    first = [shape for shape in planned if shape.refines is None]
    drawn = mikro.create_annotations(
        collection=collection.id,
        annotations=[_spec(shape, traced_on, originals) for shape in first],
    )
    originals.update(
        {
            shape.snt_id: annotation.id
            for shape, annotation in zip(first, drawn, strict=True)
        }
    )

    fitted = [shape for shape in planned if shape.refines is not None]
    if fitted:
        mikro.create_annotations(
            collection=collection.id,
            annotations=[_spec(shape, traced_on, originals) for shape in fitted],
        )

    task.progress(100, f"Imported {len(planned)} shape(s)")
    return collection


if __name__ == "__main__":
    run(app)
