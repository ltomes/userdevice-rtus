# Naming

    userdevice-<family>-<version>-<shape>

- **`userdevice`** — the MAKE. Fixed; names the suite, not this model.
- **`<family>`** — **`rtus`**. See the decision below.
- **`<version>`** — semver. Release 0.0.1 shipped as `userdevice-0.0.1-*`,
  before the family slot existed; read that stem as legacy make+version.
- **`<shape>`** — input tier: `540p`, `720p`, `1080p`.

Written **RTUS** in prose, `rtus` in artifact stems and paths — matching
every other stem in the project, which is lowercase.

## The decision: RTUS — Real Time Upscale

The family is named for the thing that distinguishes it: it targets a
**frame rate on named hardware**. Most super-resolution models optimise a
quality metric and never report a frame time at all; this line treats the
frame budget as the constraint and spends quality inside it. The name says
so out loud.

The positioning that follows from the name, and that should appear wherever
the model is described:

1. Name the **target hardware** — Jetson Thor class, TensorRT FP16.
2. State the **achieved throughput** at each shape, from measurement.
3. Say that real time **is the goal**, not a side effect — and that most
   competing models do not target it at all.

### `rtus` over `rtu`

`RTU` is a heavily-used initialism elsewhere: Remote Terminal Unit in SCADA
and industrial control, and Modbus RTU. `rtus` keeps the meaning without
landing on that collision.

### Why this does not violate the earlier "not a description" rule

An earlier draft of this file argued for a name rather than a description.
That was aimed at candidates like "upscale" — a capability **both** model
lines share, which therefore distinguishes nothing. `rtus` names the
*constraint* this line is built around, and the face-restoration line does
not share it: that line has a large per-pixel budget and optimises identity
preservation, not throughput. The two stay distinct.

## What the family name has to do

| | This line (RTUS) | Face line |
|---|---|---|
| Touches | every pixel, whole frame | a detected region only |
| Budget | hard 38 ms real-time, Thor class | large per-pixel budget |
| Constraint | throughput | identity preservation |

## Rejected

**Halide** (collides with halide-lang), **Aperture** (Valve),
**Reel / Telecine / Nitrate / Emulsion** (they name film, and the training
content turned out to be photographs — see `data/README.md`), **Kestrel**
and the small-raptor convention, and bare **upscale** (shared by both lines,
so it distinguishes nothing).
