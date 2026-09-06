"""Render GitHub release notes from a release's verdict file.

    python scripts/render_release_notes.py v0.1.0 [--out NOTES.md]

WHY THIS EXISTS RATHER THAN A HAND-WRITTEN RELEASE BODY. The model card is a
LIVING document -- it is edited whenever the prose, the licence or the
limitations change. A release note is the opposite: it is a point-in-time
record of one checkpoint, and nobody ever goes back to correct one. Pasting
the card into a release would therefore guarantee a contradiction later, and
this project has already had the card go stale in both places it exists.

So the split is by mutability, not by topic. What goes in a release note is
what can never legitimately change for that tag: the measurements. A different
number would mean a different artifact. Everything that can be re-decided --
licence terms, the trademark policy, limitations, framing -- stays in the card
and is linked, never copied.

The numbers come from releases/<tag>/verdict.json, which the cluster's analyse
Job produces, so they are transcribed by a machine rather than retyped by a
human into a workflow file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = "ltomes/userdevice-rtus"
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    """Exit non-zero and loudly. A release built on a wrong file is worse than
    no release, because it publishes numbers nobody measured."""
    print(f"ABORT: {msg}", file=sys.stderr)
    raise SystemExit(1)


def load(tag: str) -> dict:
    version = tag[1:] if tag.startswith("v") else tag
    path = ROOT / "releases" / f"v{version}" / "verdict.json"
    if not path.is_file():
        _fail(
            f"{path.relative_to(ROOT)} does not exist.\n"
            "Every tag needs its verdict file: it is the only machine-readable\n"
            "record of what the four-leg gate measured for that checkpoint."
        )
    data = json.loads(path.read_text())

    # The tag is the provenance anchor the model card points at, so a
    # disagreement here means the card would cite numbers from a different run.
    if data.get("version") != version:
        _fail(
            f"version mismatch: tag is {version!r}, "
            f"{path.name} declares {data.get('version')!r}"
        )
    return data


def check_pyproject(version: str) -> None:
    """The version is maintained in several places by hand. Rather than add a
    fourth, make disagreement a build failure."""
    text = (ROOT / "pyproject.toml").read_text()
    if f'version = "{version}"' not in text:
        _fail(
            f"pyproject.toml does not declare version = \"{version}\".\n"
            "The tag, the verdict file and the package must agree."
        )


def render(d: dict) -> str:
    v = d["version"]
    ck = d["checkpoint"]
    sel = d["selection"]
    tp = d["throughput"]
    par = d["parity"]
    w = d["weights"]

    out: list[str] = []
    add = out.append

    add(f"**{ck['tier']} tier, stage-{ck['stage']} checkpoint at iteration "
        f"{ck['iteration']:,}** — architecture `{ck['architecture']}`, "
        f"{ck['parameters']:,} parameters, {ck['scale']}x.")
    add("")
    # Fixed precision, not repr(). JSON drops trailing zeros, so a bare
    # {value} renders 0.14194 where the model card publishes 0.141940 -- a
    # release note disagreeing with the card about a headline metric is the
    # exact drift this whole file exists to prevent.
    add(f"Selected best-by-{sel['metric']} at **{sel['value']:.6f}** across "
        f"{sel['candidates']} candidates.")
    add("")

    add("## Acceptance — four legs, on identical frames")
    add("")
    add("A checkpoint is accepted by passing every leg, not because one number "
        "improved.")
    add("")
    add("| Leg | Result | Verdict |")
    add("|---|---|---|")
    for leg in d["legs"]:
        verdict = leg["verdict"]
        if leg.get("blocking"):
            verdict = f"**{verdict}** (blocking)"
        elif verdict == "PASS":
            verdict = f"**{verdict}**"
        add(f"| {leg['name']} | {leg['result']} | {verdict} |")
    add("")

    add("## Measured throughput")
    add("")
    # No .capitalize(): it lowercases everything after the first character,
    # which turns "median enqueueV3 GPU time" into "enqueuev3 gpu time".
    add(f"{tp['device']} — {tp['runtime']}. {tp['method'][:1].upper()}"
        f"{tp['method'][1:]}.")
    add("")
    add("| Input | Output | Latency (median) | Throughput |")
    add("|---|---|---|---|")
    for s in tp["shapes"]:
        # Three decimals to match the card: 12.350, not 12.35.
        add(f"| {s['input']} | {s['output']} | {s['latency_ms']:.3f} ms | "
            f"{s['fps']} fps |")
    add("")
    add(f"{tp['note']}. These are compute-only figures on one named device; "
        "they are not a promise about your pipeline or end-to-end playback.")
    add("")
    add(f"TensorRT parity for this checkpoint: **{par['db']} dB** against the "
        f"{par['against']} (project threshold >{par['threshold_db']} dB).")
    add("")

    add("## Weights")
    add("")
    add(f"The weights are **not attached to this release**. They are published "
        f"on {w['host']} at [`{w['repo']}`](https://huggingface.co/{w['repo']})"
        f", revision `{w['revision']}`, with `SHA256SUMS` and the sweep, "
        "temporal and invention evidence beside them.")
    add("")
    add("Keeping one home for the artifacts is deliberate: two copies of a "
        "weights file is two things to verify and one of them will drift.")
    add("")

    add("## Full model card")
    add("")
    card = f"https://github.com/{REPO}/blob/v{v}/docs/MODEL-CARD.md"
    add("Licence, attribution, training data, degradation recipe and "
        f"limitations are in [`docs/MODEL-CARD.md`]({card}) at this tag. "
        "They are deliberately not copied here — the card is edited over "
        "time and a release note is not.")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", help="release tag, e.g. v0.1.0")
    ap.add_argument("--out", help="write here instead of stdout")
    args = ap.parse_args()

    data = load(args.tag)
    check_pyproject(data["version"])
    body = render(data)

    if args.out:
        pathlib.Path(args.out).write_text(body)
        print(f"wrote {args.out} ({len(body)} bytes)")
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
