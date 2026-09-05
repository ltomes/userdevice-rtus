"""Which frames do we handle worst — and worse than the teacher?

An average hides the failures that matter. A model can clear the ship bar on
the mean while being clearly beaten on a whole class of content, and that
class is where the next dataset or loss decision should come from.

Consumes the --per-image JSON that eval_compare.py writes, and reports:

  WORST ABSOLUTE   frames where our metric is poor outright — usually hard
                   content (heavy grain, motion blur), not necessarily a
                   model failure.
  WORST vs TEACHER frames where the teacher beats us by the widest margin.
                   THIS is the actionable list: the teacher proves the
                   detail was recoverable, and we did not recover it.
  BEST vs TEACHER  where we win hardest — worth knowing so a texture push
                   does not regress what already works.

Usage (inside the container image, cwd /workspace):
  python analyze_outliers.py results/x-perimage.json \
      --ours iter050000 --metric dists [--top 12]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("per_image")
    ap.add_argument("--ours", required=True, help="label of our candidate")
    ap.add_argument("--vs", default="teacher_web")
    ap.add_argument("--metric", default="dists",
                    choices=["dists", "lpips", "psnr", "ssim"])
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    with open(args.per_image) as f:
        data = json.load(f)

    lower_better = args.metric in ("dists", "lpips")
    rows = []
    for frame, cands in data.items():
        if args.ours not in cands or args.vs not in cands:
            continue
        o = cands[args.ours][args.metric]
        t = cands[args.vs][args.metric]
        # positive delta always means "we are worse", whichever direction
        # the metric runs, so one sort works for both kinds
        delta = (o - t) if lower_better else (t - o)
        rows.append((frame, o, t, delta))

    if not rows:
        print(f"no frames with both '{args.ours}' and '{args.vs}'")
        print(f"available labels: "
              f"{sorted(next(iter(data.values())).keys())}")
        sys.exit(1)

    n = len(rows)
    losses = [r for r in rows if r[3] > 0]
    print(f"{n} frames scored on {args.metric}. "
          f"We are beaten by {args.vs} on {len(losses)} "
          f"({len(losses) / n * 100:.0f}%).")

    def table(title, sel, note):
        print(f"\n{title}\n{note}")
        print(f"| frame | ours | {args.vs} | delta |")
        print("|---|---|---|---|")
        for frame, o, t, d in sel:
            print(f"| {frame} | {o:.4f} | {t:.4f} | {d:+.4f} |")

    worst_abs = sorted(rows, key=lambda r: -r[1] if lower_better else r[1])
    table(f"WORST ABSOLUTE {args.metric}",
          worst_abs[:args.top],
          "Hard content. Not necessarily our failure — check the teacher "
          "column; if it is also poor, the frame is just difficult.")

    table(f"WORST vs {args.vs} — THE ACTIONABLE LIST",
          sorted(rows, key=lambda r: -r[3])[:args.top],
          "The teacher recovered detail here and we did not. These frames "
          "are where the next improvement lives.")

    table(f"BEST vs {args.vs}",
          sorted(rows, key=lambda r: r[3])[:args.top],
          "Our strongest frames — a texture push must not regress these.")


if __name__ == "__main__":
    main()
