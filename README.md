# snt-traces-importer

<!-- arkitekt:badge -->
[![Open in Arkitekt](https://arkitekt.live/img/badge/open-in-arkitekt.svg)](https://arkitekt.live/deeplink?orkestrator=%2Fdo%2Fv1%2Frepo.install%3Fsource%3Doci%253Aghcr.io%252Farkitektio%252Fsnt-traces-importer)
<!-- /arkitekt:badge -->

An Arkitekt plugin that imports SNT (Simple Neurite Tracer) `.traces` files into mikro as
annotation collections.

**Import SNT Traces** (`import_snt_traces(file, image, flip_z)`) takes the `.traces` file and
the image it was traced on. It creates one annotation collection on that image:

- **One shape per path.** Each SNT path becomes a PATH annotation, or a POINT if SNT kept a
  single vertex. Vertices are the image's level-0 pixel indices, in its axis order (`z, y, x`).
- **Channel as `tracedOn`.** The channel a path was traced on is stored as `tracedOn`: the
  lens of that channel, metadata only. It is not a `c` pin.
- **Frame as a pin.** A path's frame pins `t`, when the image has a `t` axis.
- **Fitted copies.** SNT's fitted copy of a path is drawn as a refinement of it
  (`derivedFrom … REFINED`).
- **Branch topology.** Which path branches off which, and at which vertex, is written into
  each annotation's description.

The import refuses before writing anything when:

- the file's image size differs from the image's;
- a vertex lies outside the image;
- a path names a channel or frame the image does not have.

If the voxel sizes disagree, it logs a warning and imports anyway.

`flip_z` has no default on purpose. The ND2 converter turns top-down stacks over, and when
it does it notes "z-order: reversed from top-down" on the image. Fiji traced the file's own
order, so set `flip_z` for those images.

Needs mikro ≥ 13 (`Annotation.tracedOn`).

```
uv run pytest      # offline: the parser and the planning, no server
arkitekt run dev
```
