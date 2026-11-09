"""A town day with trips that have a purpose, in one picture (docs/img/town05_day.png), from a run of
tools/check_purpose_day.py (OUT/purpose_seed<seed>.json):

* the trips leaving each hour, by why (to work, shopping, eating out, visitors, the rest);
* the cars on the roads and the cars parked, over the day;
* each car park filling and emptying (how full, every 10 town minutes).

    python tools/purpose_day_chart.py out/purpose_day_train/purpose_seed101.json docs/img/town05_day.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SURFACE, PANEL, GRID = "#0d111a", "#121826", "#1f2738"
TEXT, MUTED = "#eef2fa", "#97a3bd"
# the categorical slots in their fixed order (dark steps, checked with the dataviz validator on this surface)
GROUPS = [("To work", ("work",), "#3987e5"), ("Shopping", ("shopping",), "#d95926"),
          ("Eating out", ("eating",), "#199e70"), ("Visitors", ("visitor",), "#c98500"),
          ("Park, hospital, errands, deliveries", ("park", "care", "errand", "deliver"), "#d55181")]
SEQ = ["#0d111a", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]   # empty -> full


def main(src: str, dst: str) -> int:
    r = json.loads(Path(src).read_text(encoding="utf-8"))
    card = json.loads((ROOT / "scenarios" / "town05" / "places.json").read_text(encoding="utf-8"))
    name, cap = {}, {}
    for p in card["places"]:
        for a in [p.get("parking")] + [k["id"] for k in p.get("kerbs") or ()]:
            if a:
                name[a] = p["name"]
    for a in card["parking_areas"]:
        cap[a["id"]] = len(a["spaces"]) if "spaces" in a else int(a.get("roadside", 0))
    by_hour = r["purposes"]["by_hour"]
    start = int(r["start"])
    hours = [(start + i) % 24 for i in range(24)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(16, 11.2), dpi=110, facecolor=SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.8, 1.5], hspace=0.42, left=0.15, right=0.97, top=0.88,
                          bottom=0.06)
    fig.text(0.15, 0.955, "TOWN05  ·  A DAY OF TRIPS WITH A PURPOSE", fontsize=22, weight="bold", color=TEXT)
    fig.text(0.15, 0.925, f"SUMO, measured: {r['hours']:.0f} town hours from {start:02d}:00, seed {r['seed']}  ·  "
                          f"{r['legs']} trips ended in a parking space  ·  {r['purposes']['through']} through-traffic "
                          f"trips  ·  {r['stuck_over_300s']} cars stuck over 5 min", fontsize=12, color=MUTED)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=14, color=TEXT, pad=10, weight="bold")

    # 1. the trips leaving each hour, by why
    ax = fig.add_subplot(gs[0])
    style(ax, "Trips leaving each hour, by why")
    bottom = [0.0] * 24
    for label, keys, col in GROUPS:
        vals = [sum(by_hour.get(str(h), {}).get(k, 0) for k in keys) for h in hours]
        ax.bar(range(24), vals, bottom=bottom, color=col, width=0.78, label=label, edgecolor=PANEL, linewidth=1.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(24), [f"{h:02d}" for h in hours])
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylabel("trips")
    leg = ax.legend(ncol=5, loc="upper right", frameon=False, fontsize=11, bbox_to_anchor=(1.0, 1.2))
    for t in leg.get_texts():
        t.set_color(TEXT)
    peak = max(range(24), key=lambda i: bottom[i])
    ax.annotate(f"busiest hour: {bottom[peak]:.0f} trips", (peak, bottom[peak]), (peak + 1.2, bottom[peak] * 0.92),
                color=TEXT, fontsize=11, arrowprops={"arrowstyle": "-", "color": MUTED})

    # 2. on the roads and parked
    s = r["series"]
    x = [i / 6.0 for i in range(len(s))]
    ax = fig.add_subplot(gs[1])
    style(ax, "Cars on the roads and cars parked")
    parked = [sum(p["occ"].values()) for p in s]
    moving = [p["moving"] for p in s]
    ax.plot(x, moving, color="#3987e5", lw=2)
    ax.plot(x, parked, color="#d95926", lw=2)
    ax.text(x[-1] + 0.15, moving[-1], "on the roads", color=TEXT, va="center", fontsize=11)
    ax.text(x[-1] + 0.15, parked[-1], "parked", color=TEXT, va="center", fontsize=11)
    ax.set_xticks(range(24), [f"{h:02d}" for h in hours])
    ax.set_xlim(-0.1, 25.4)
    ax.set_ylabel("cars")

    # 3. each car park, how full (a place's street spaces on both sides together)
    groups: dict = {}
    for a in sorted({a for p in s for a in p["occ"]} | set(cap)):
        if cap.get(a):
            groups.setdefault(name.get(a, a), []).append(a)
    room = {g: sum(cap[a] for a in aa) for g, aa in groups.items()}
    fill = {g: [min(1.0, sum(p["occ"].get(a, 0) for a in aa) / room[g]) for p in s] for g, aa in groups.items()}
    street = {g for g, aa in groups.items() if all(a.startswith("pa_kerb") for a in aa)}
    rows = sorted((g for g in groups if max(fill[g]) > 0), key=lambda g: (g in street, -max(fill[g])))
    grid = [fill[g] for g in rows]
    ax = fig.add_subplot(gs[2])
    style(ax, "How full each car park is (every 10 town minutes)")
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("fill", SEQ)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, len(s) / 6.0, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows)), [f"{g}  ({room[g]})" + ("  · street" if g in street else "") for g in rows],
                  fontsize=10.5)
    ax.set_xticks(range(24), [f"{h:02d}" for h in hours])
    for i in range(1, len(rows)):
        ax.axhline(i - 0.5, color=SURFACE, lw=1.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["empty", "half", "full"], color=MUTED)
    cb.outline.set_visible(False)
    fig.savefig(dst, facecolor=SURFACE)
    print("wrote", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
