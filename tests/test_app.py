"""The import with no server: a small SNT file in, the shapes that would be drawn out."""

import gzip

import pytest

from app import plan_annotations
from snt_traces import read_tracings

TRACES = """<?xml version="1.0" encoding="UTF-8"?>
<tracings>
  <samplespacing x="0.2" y="0.2" z="0.5" units="µm"/>
  <imagesize width="16" height="8" depth="4"/>
  <path id="1" swctype="0" color="None" channel="2" frame="1" name="Axon">
    <point x="1" y="2" z="0"/>
    <point x="3" y="4" z="1"/>
  </path>
  <path id="2" swctype="3" color="#ff0000" channel="1" frame="1" startson="1" startsindex="1" name="Branch">
    <point x="5" y="6" z="3"/>
  </path>
  <path id="3" swctype="0" color="None" channel="2" frame="1" fittedversionof="1" name="Axon (fitted)">
    <point x="1.1" y="2.1" z="0"/>
    <point x="3.1" y="4.1" z="1"/>
  </path>
</tracings>
"""


def test_a_tracing_becomes_shapes_on_its_image(tmp_path):
    path = tmp_path / "cell.traces"
    path.write_bytes(gzip.compress(TRACES.encode()))
    tracings = read_tracings(path)

    axon, branch, fitted = plan_annotations(tracings, ("c", "z", "y", "x"), (2, 4, 8, 16), flip_z=False)

    # SNT's (x, y, z) becomes the image's (z, y, x); channel 2 is lens c=1, never a pin.
    assert axon.vectors == [[0, 2, 1], [1, 4, 3]]
    assert (axon.kind.value, axon.channel, axon.t, axon.stroke_color) == ("PATH", 1, None, None)
    assert (branch.kind.value, branch.channel, branch.stroke_color) == ("POINT", 0, [255, 0, 0, 255])
    assert "branches off Axon at its vertex 1" in branch.description
    assert fitted.refines == 1

    flipped, _, _ = plan_annotations(tracings, ("c", "z", "y", "x"), (2, 4, 8, 16), flip_z=True)
    assert [vertex[0] for vertex in flipped.vectors] == [3, 2]

    with pytest.raises(ValueError, match="traced on another image"):
        plan_annotations(tracings, ("c", "z", "y", "x"), (2, 5, 8, 16), flip_z=False)
    with pytest.raises(ValueError, match="traced on channel 2 and the image has 1"):
        plan_annotations(tracings, ("c", "z", "y", "x"), (1, 4, 8, 16), flip_z=False)
