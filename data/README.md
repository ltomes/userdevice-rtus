# Data

**No training or test data is stored in this repository, ever.** This
directory holds the *contract*: what the data must look like and how to
regenerate it. `.gitignore` excludes everything here except this file and
the `*.example.yml` templates.

## Training corpus

The corpus is fully reproducible from public sources. Nothing private is
involved and nothing needs to be redistributed by us.

**HR source** — the NomosRealWeb release (`hr.tar`, 6000 x 512^2 Nomos-v2
high-resolution images). Public.

**LR** — generated locally from those HRs. There are two generations:

| Corpus | Script | Degradation |
|---|---|---|
| v1 (`greyduck2x`) | `scripts/gen_pairs_h264.sh` | downscale 1/2 (random filter: bicubic/bilinear/lanczos/area) -> libx264 CRF 20-32, yuv420p -> decode. 5700 train / 300 val. |
| v2 (`greyduck2x_v2`) | `scripts/gen_pairs_v2.py` | HR motion first (sub-pixel drift/zoom over a 480px window, so the LR = downscale(HR) contract stays pixel-exact), then downscale 1/2, then libx264 CRF 18-34 with GOP {24,48} and scenecut=0, harvesting P/B frames that carry reference drift and inter deblocking. |

Two properties of this recipe are load-bearing and should not be "cleaned
up" by a later contributor:

1. **Motion is applied on the HR path, never the LR path.** Applying it to
   LR breaks the pairing. This was a real defect in an earlier naive
   zoompan recipe.
2. **There is no synthetic noise stage.** HR and pre-encode LR carry the
   same photo-native grain; only the codec degrades the LR. This is a
   deliberate grain-preservation rule -- the student must not learn to
   denoise, because grain is signal in the target domain.

**Teacher targets** — `scripts/gen_teacher_targets.py` runs
`4xNomosWebPhoto_RealPLKSR` over the LR and downscales its output 2x
(Lanczos) to the student's GT size. The downscale itself attenuates GAN
hallucination. The teacher was trained with noise injection and therefore
denoises, which is why the late stage anneals back onto real GT to restore
grain.

## What this model was NOT trained on

It was not trained on film, television, or any private media library. The
`_film` in the config names denotes a **latency tier** -- the 38 ms budget
of a 24fps frame -- not the content. Motion in the corpus is synthetic and
the HR frames are photographs. Any claim to the contrary in a model card
would be false.

## Mounting

Configs address data at container-relative paths (`/workspace/datasets/...`),
so the host layout is yours to choose. Copy `data/datasets.example.yml` to
`data/datasets.local.yml` (gitignored) and mount accordingly.

## Test data

See `docs/BENCHMARKS.md`. Test sets are fetched by
`scripts/fetch_testsets.sh` into `testsets/` and are never committed.
