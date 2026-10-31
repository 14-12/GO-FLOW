"""How fast the see-through overlay window updates on this PC (Windows), without CARLA.

    python tools/bench_overlay.py

Opens a borderless window over the whole primary screen, styled as the free-fly overlay (cosim/overlay.py:
layered, colour key, click-through, on top), draws the native camera feed's labels and boxes on it
(cosim/cctv.CCTV.draw_native) and times, 60 redraws each, three ways of sending them to the window:
  rects  the rectangles drawn (and those of the redraw before), as the viewer did (about 25 a redraw)
  bands  the same merged into horizontal bands (top labels, boxes, bottom labels)
  flip   the whole window
Writes out/freefly/bench_overlay.json and closes (about 15 s).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bands(rects, w: int, gap: int = 40) -> list:
    """Rectangles merged into full-width horizontal bands (rows within `gap` pixels joined)."""
    import pygame

    spans = sorted((r.top, r.bottom) for r in rects if r.width > 0 and r.height > 0)
    out: list[list[int]] = []
    for top, bottom in spans:
        if out and top <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], bottom)
        else:
            out.append([top, bottom])
    return [pygame.Rect(0, t, w, b - t) for t, b in out]


def main() -> int:
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
    import pygame

    from cosim.cctv import CCTV
    from cosim.overlay import KEY_COLOUR, Overlay, dpi_aware

    dpi_aware()
    pygame.init()
    info = pygame.display.Info()
    size = (info.current_w, info.current_h)
    screen = pygame.display.set_mode(size, pygame.NOFRAME)
    ov = Overlay(pygame.display.get_wm_info().get("window", 0))
    screen.fill(KEY_COLOUR)
    pygame.display.flip()
    cctv = CCTV()
    view = (None, {"vehicles": 6, "pedestrians": 2}, 12.0, "979-4 looking west")
    w, h = size
    res = {"screen": list(size), "overlay_ok": ov.ok}
    for mode in ("rects", "bands", "flip"):
        prev: list = []
        draw_ms = send_ms = 0.0
        t_all = time.perf_counter()
        for k in range(60):
            for e in pygame.event.get():
                pass
            dx = 8 * k
            boxes = [(300 + dx, 400, 420 + dx, 480, "car", 0.9), (900 - dx, 350, 1000 - dx, 420, "car", 0.8),
                     (600, 500 + k, 640, 580 + k, "pedestrian", 0.7)]
            t0 = time.perf_counter()
            for r in prev:
                screen.fill(KEY_COLOUR, r)
            rects = cctv.draw_native(screen, size, view, "979-4 looking west", "detector", "MON  ·  DAY 1  ·  12:00",
                                     boxes, now=k / 30)
            t1 = time.perf_counter()
            if mode == "rects":
                pygame.display.update(rects + prev)
            elif mode == "bands":
                pygame.display.update(bands(rects + prev, w))
            else:
                pygame.display.flip()
            t2 = time.perf_counter()
            draw_ms += 1000 * (t1 - t0)
            send_ms += 1000 * (t2 - t1)
            prev = rects
        span = time.perf_counter() - t_all
        res[mode] = {"redraws_per_s": round(60 / span, 1), "draw_ms": round(draw_ms / 60, 2),
                     "send_ms": round(send_ms / 60, 2), "rects_per_redraw": len(rects)}
        print(mode, res[mode], flush=True)
    pygame.quit()
    out = ROOT / "out" / "freefly" / "bench_overlay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
