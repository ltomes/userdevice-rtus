# Naming

The scheme has three slots:

    userdevice-<family>-<version>-<shape>

- **`userdevice`** — the MAKE. Fixed. Names the suite, not this model.
- **`<family>`** — the model family. **NOT YET CHOSEN**, tracked in this
  repo as the literal token `FAMILY_NAME_PENDING`. Run
  `scripts/set-family-name.sh <name>` to stamp it everywhere at once.
- **`<version>`** — semver. Release 0.0.1 shipped as `userdevice-0.0.1-*`,
  before the family slot existed; read that stem as legacy make+version.
- **`<shape>`** — input tier: `540p`, `720p`, `1080p`.

## What the family name has to do

It has to distinguish this line from the face-restoration line. Both models
upscale, so **the family name cannot be "upscale"** or any word the two
share — that would leave the other line no name to take, and would imply it
is a variant of this one when they are separate lines with different
architectures, teachers and budgets.

The axis that actually separates them is scope and budget:

| | This line | Face line |
|---|---|---|
| Touches | every pixel, whole frame | a detected region only |
| Budget | hard 38 ms real-time, Thor-class hardware | large per-pixel budget |
| Constraint | speed | identity preservation |

## Constraints on the choice

- One word. A name, not a description.
- No "upscale", "SR", "video", or "2x" in it — the make and version already
  say it is a model, and the shape suffix says what it takes.
- Must leave a sibling name available for the face line.

Rejected so far: **Halide** (collides with halide-lang), **Aperture**
(Valve), **Reel/Telecine/Nitrate/Emulsion** (they name film, and the
training content turned out to be photographs — see `data/README.md`),
**Kestrel** and the small-raptor convention (operator declined).
