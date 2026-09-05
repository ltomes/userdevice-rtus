"""Build a self-contained HTML report from eval results + A/B panels.

Why this exists: the panels and JSON live on ms, and reviewing them means
being at the media centre. This bakes everything into one HTML file with
images inlined as data URIs, so the report can be opened anywhere with no
server, no mounts, and no missing-image boxes.

Panels are re-encoded to JPEG and width-capped, because a 4-column PNG panel
is several MB and a dozen of them will not fit in a single page.

Usage (inside the container image, cwd /workspace):
  python -m userdevice_rtus.tools.make_report --metrics results/x.json --panels visual_ab_30k \
      --title "stage-C @30k" --out /workspace/report.html
"""
import argparse
import base64
import glob
import io
import json
import os
import re

from PIL import Image

MAX_W = 1600
JPEG_Q = 86
# lower-is-better metrics, so the table can mark winners correctly
LOWER_BETTER = {"dists", "lpips"}
OURS_RE = re.compile(r"stage|ours|iter", re.I)

NICE = {"psnr": "PSNR", "ssim": "SSIM", "dists": "DISTS", "lpips": "LPIPS"}


def img_data_uri(path):
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_Q, optimize=True)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode()), im.width, im.height


def verdict_cards(metrics):
    """The headline: how close our student is to the web model.

    Reported as a percentage of the teacher's score because the absolute
    DISTS/LPIPS numbers mean nothing to anyone who does not work with them
    daily, whereas "within 3%" is immediately legible.
    """
    ours = next((k for k in metrics
                 if OURS_RE.search(k) and "stageA" not in k), None)
    teach = next((k for k in metrics if "teacher" in k), None)
    if not (ours and teach):
        return ""
    o, t = metrics[ours], metrics[teach]
    cards = []
    for m in ("dists", "lpips"):
        if m in o and m in t:
            gap = (o[m] - t[m]) / t[m] * 100
            cards.append(
                (NICE[m] + " vs web model",
                 f"{gap:+.1f}%",
                 "perceptual distance — negative means we are closer to the "
                 "original than the web model is",
                 gap <= 2.0))
    if "psnr" in o and "psnr" in t:
        d = o["psnr"] - t["psnr"]
        cards.append(("Fidelity vs web model", f"{d:+.2f} dB",
                      "PSNR — positive means we invent less detail that was "
                      "never in the source", d > 0))
    html = ["<div class=verdict>"]
    for k, v, d, good in cards:
        cls = " good" if good else ""
        html.append(f"<div class=stat><div class=k>{k}</div>"
                    f"<div class='v{cls}'>{v}</div><div class=d>{d}</div></div>")
    html.append("</div>")
    return "".join(html)


def probe_table(probe):
    """HF recovery vs invention rate, read against the bicubic control.

    Invention rate is only meaningful as a DELTA against bicubic: bicubic
    cannot invent anything, so whatever it scores is the detector's
    false-positive floor. Absolute percentages here would mislead.
    """
    if not probe:
        return ""
    floor = probe.get("bicubic", {}).get("invention_rate")
    rows = []
    order = (["bicubic"]
             + [k for k in probe if k not in ("bicubic", "teacher_web")]
             + ["teacher_web"])
    for k in order:
        if k not in probe:
            continue
        v = probe[k]
        rec = v.get("hf_recovery")
        inv = v.get("invention_rate")
        if k == "bicubic":
            excess = "<span class='muted'>control &mdash; floor</span>"
        elif floor is not None and inv is not None:
            e = (inv - floor) * 100
            cls = "win" if e <= 0 else ""
            excess = (f"<span class='{cls}'>{e:+.2f} pts</span>" if cls
                      else f"{e:+.2f} pts")
        else:
            excess = "&mdash;"
        ours = " class='ours'" if OURS_RE.search(k) else ""
        rows.append(
            f"<tr{ours}><th class='row'>{k}</th>"
            f"<td>{rec:.3f}&times;</td><td>{inv * 100:.2f}%</td>"
            f"<td>{excess}</td>"
            f"<td class='n'>{v.get('frames', '')}</td></tr>")
    return ("<div class=tablewrap><table><thead><tr>"
            "<th class='row'>candidate</th><th>HF recovery &uarr;</th>"
            "<th>invention rate</th><th>excess vs control &darr;</th>"
            "<th class='n'>frames</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def metric_table(metrics):
    if not metrics:
        return "<p>No metrics supplied.</p>"
    cols = [c for c in ("psnr", "ssim", "dists", "lpips")
            if any(c in v for v in metrics.values())]
    best = {}
    for c in cols:
        vals = {k: v[c] for k, v in metrics.items() if c in v}
        best[c] = (min(vals, key=vals.get) if c in LOWER_BETTER
                   else max(vals, key=vals.get))
    head = "".join(
        f"<th>{NICE[c]} {'&darr;' if c in LOWER_BETTER else '&uarr;'}</th>"
        for c in cols)
    rows = []
    for label, v in metrics.items():
        cells = []
        for c in cols:
            if c not in v:
                cells.append("<td>&mdash;</td>")
                continue
            win = " class='win'" if best[c] == label else ""
            cells.append(f"<td{win}>{v[c]:.4f}</td>")
        ours = " class='ours'" if OURS_RE.search(label) else ""
        rows.append(f"<tr{ours}><th class='row'>{label}</th>"
                    f"{''.join(cells)}"
                    f"<td class='n'>{v.get('n', '')}</td></tr>")
    return (f"<table><thead><tr><th class='row'>candidate</th>{head}"
            f"<th class='n'>n</th></tr></thead><tbody>"
            f"{''.join(rows)}</tbody></table>")


FONTS = ("<link rel='preconnect' href='https://fonts.googleapis.com'>"
         "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
         "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
         "family=Fraunces:opsz,wght@9..144,500;9..144,700&"
         "family=IBM+Plex+Mono:wght@400;600&"
         "family=IBM+Plex+Sans:wght@400;500;600&display=swap'>")

CSS = """
:root{
  --bg:#fcfcfd; --panel:#f2f3f6; --fg:#14171c; --muted:#5d6470;
  --line:#dfe2e9; --accent:#b8781f; --accent-soft:#f4e7d1;
  --win:#12703a; --win-bg:#e4f3ea; --shadow:0 1px 2px rgba(16,20,28,.06);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0e1116; --panel:#161b22; --fg:#e9ecf2; --muted:#98a1b0;
  --line:#242b36; --accent:#e3a94e; --accent-soft:#3a2c14;
  --win:#5fd98d; --win-bg:#123024; --shadow:0 1px 2px rgba(0,0,0,.4);
}}
:root[data-theme="dark"]{
  --bg:#0e1116; --panel:#161b22; --fg:#e9ecf2; --muted:#98a1b0;
  --line:#242b36; --accent:#e3a94e; --accent-soft:#3a2c14;
  --win:#5fd98d; --win-bg:#123024; --shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{margin:0;padding:40px 24px 72px;background:var(--bg);color:var(--fg);
  font:16px/1.6 "IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1680px;margin:0 auto}
.eyebrow{font:600 12px/1 "IBM Plex Mono",ui-monospace,monospace;
  letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
  margin:0 0 14px}
h1{font:700 clamp(30px,4vw,46px)/1.08 Fraunces,Georgia,serif;
  margin:0 0 12px;letter-spacing:-.015em;text-wrap:balance}
.lede{margin:0 0 8px;max-width:66ch;color:var(--muted);font-size:17px}
h2{font:600 22px/1.2 Fraunces,Georgia,serif;margin:52px 0 6px;
  letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 18px;max-width:74ch;font-size:14.5px}
.verdict{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,
  minmax(230px,1fr));margin:30px 0 8px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:18px 20px;box-shadow:var(--shadow)}
.stat .k{font:600 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
.stat .v{font:600 30px/1.15 "IBM Plex Mono",monospace;margin:10px 0 4px;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .v.good{color:var(--win)}
.stat .d{font-size:13.5px;color:var(--muted);line-height:1.45}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;
  background:var(--panel);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;
  font-variant-numeric:tabular-nums}
th,td{padding:11px 18px;text-align:right;border-bottom:1px solid var(--line);
  white-space:nowrap;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:14px}
thead th{font:600 11.5px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}
th.row{text-align:left;font-family:"IBM Plex Sans",sans-serif;font-size:14.5px;
  font-weight:600;letter-spacing:0;text-transform:none}
tbody th.row{color:var(--fg)}
td.win{color:var(--win);background:var(--win-bg);font-weight:600}
td.n{color:var(--muted)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tbody tr.ours th.row::after{content:"ours";margin-left:9px;font:600 10px/1
  "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent);background:var(--accent-soft);padding:3px 6px;
  border-radius:4px;vertical-align:2px}
figure{margin:0 0 28px;border:1px solid var(--line);border-radius:12px;
  overflow:hidden;background:var(--panel);box-shadow:var(--shadow)}
figcaption{padding:10px 18px;color:var(--muted);
  font:400 12.5px/1.4 "IBM Plex Mono",monospace;
  border-bottom:1px solid var(--line)}
figure img{display:block;width:100%;height:auto}
.note{background:var(--panel);border-left:3px solid var(--accent);
  padding:14px 18px;border-radius:0 10px 10px 0;margin:22px 0;
  font-size:14.5px;max-width:78ch}
.note b{color:var(--fg)}
ul.caveats{max-width:78ch;color:var(--muted);font-size:14.5px;
  padding-left:20px}
ul.caveats li{margin:6px 0}
.muted{color:var(--muted)}
span.win{color:var(--win);font-weight:600}
code{font:400 13px/1 "IBM Plex Mono",monospace;background:var(--accent-soft);
  color:var(--fg);padding:2px 6px;border-radius:4px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--panels", default=None)
    ap.add_argument("--probe", default=None,
                    help="hallucination_probe.py JSON")
    ap.add_argument("--panels-b", default=None,
                    help="a SECOND panel set, rendered after the first — "
                         "for the frames we lose on, which an even sample "
                         "will usually miss entirely")
    ap.add_argument("--panels-b-title", default="Where we lose")
    ap.add_argument("--panels-b-note", default=None)
    ap.add_argument("--title", default="upscaler comparison")
    ap.add_argument("--headline", default=None)
    ap.add_argument("--eyebrow", default="userDevice models")
    ap.add_argument("--note", default=None)
    ap.add_argument("--caveats", default=None,
                    help="pipe-separated honest limitations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.headline:
        args.headline = args.title

    # Fail loudly on a missing input. Silently skipping a requested section
    # published a report that claimed to contain the invention analysis and
    # did not (2026-08-22) — the container wrote its JSON to an unmounted
    # path, so the file never existed on the host and nothing complained.
    def load(path, what):
        if not path:
            return {}
        if not os.path.exists(path):
            raise SystemExit(f"ERROR: --{what} {path} does not exist. "
                             f"If a container wrote it, check the path is on "
                             f"a MOUNTED volume — '/workspace/../results' is "
                             f"inside the container, not on the host.")
        with open(path) as f:
            return json.load(f)

    metrics = load(args.metrics, "metrics")

    parts = [f"<title>{args.title}</title>", FONTS,
             f"<style>{CSS}</style>",
             "<div class=wrap>",
             f"<p class=eyebrow>{args.eyebrow}</p>",
             f"<h1>{args.headline}</h1>",
             "<p class=sub>original vs bicubic vs our student vs the web "
             "model (teacher). Higher PSNR/SSIM = more faithful; lower "
             "DISTS/LPIPS = perceptually closer to ground truth.</p>"]
    if args.note:
        parts.append(f"<div class=note>{args.note}</div>")

    if metrics:
        parts.append(verdict_cards(metrics))
        parts += ["<h2>Scores on identical frames</h2>",
                  "<p class=sub>Best value in each column is "
                  "highlighted. PSNR and SSIM measure fidelity to the "
                  "source; DISTS and LPIPS measure how close the result "
                  "looks to a human. A texture-focused stage is expected "
                  "to trade the first pair for the second.</p>",
                  "<div class=tablewrap>",
                  metric_table(metrics), "</div>"]

    probe = load(args.probe, "probe")
    if probe:
        parts += ["<h2>Detail recovered vs detail invented</h2>",
                  "<p class=sub>HF recovery is how much of the source's "
                  "high-frequency detail came back &mdash; a 2&times; "
                  "downsample destroys detail nothing fully restores, so "
                  "every row sits below 1.0. Invention rate counts tiles "
                  "whose structure does <em>not</em> correspond to the "
                  "original. Bicubic cannot invent, so its rate is the "
                  "detector's false-positive floor and the last column, "
                  "measured against it, is the one that matters.</p>",
                  probe_table(probe)]

    if args.panels:
        files = sorted(glob.glob(os.path.join(args.panels, "*.png")))
        parts.append(f"<h2>Visual comparison ({len(files)} frames)</h2>")
        parts.append("<p class=sub>Each frame shows the full image, then a "
                     "3&times; nearest-neighbour zoom on its highest-detail "
                     "region &mdash; differences are invisible at 1:1.</p>")
        for p in files:
            uri, w, h = img_data_uri(p)
            parts.append(f"<figure><figcaption>{os.path.basename(p)}"
                         f"</figcaption><img src='{uri}' alt='comparison "
                         f"panel for {os.path.basename(p)}'></figure>")
    if args.caveats:
        items = "".join(f"<li>{c}</li>"
                        for c in args.caveats.split("|"))
        parts += ["<h2>Read this with</h2>",
                  f"<ul class=caveats>{items}</ul>"]
    if args.panels_b:
        files = sorted(glob.glob(os.path.join(args.panels_b, "*.png")))
        parts.append(f"<h2>{args.panels_b_title} ({len(files)} frames)</h2>")
        if args.panels_b_note:
            parts.append(f"<p class=sub>{args.panels_b_note}</p>")
        for p_ in files:
            uri, w, h = img_data_uri(p_)
            parts.append(f"<figure><figcaption>{os.path.basename(p_)}"
                         f"</figcaption><img src='{uri}' alt='comparison "
                         f"panel for {os.path.basename(p_)}'></figure>")
    parts.append("</div>")

    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({mb:.1f} MB)")
    if mb > 15:
        print("WARNING: >15 MB — reduce --panels count or MAX_W")


if __name__ == "__main__":
    main()
