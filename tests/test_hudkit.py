"""The overlay's design system (cosim/hudkit.py): notices, the banner, the events that make them."""

import os
from types import SimpleNamespace as NS

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402
import pytest  # noqa: E402

from cosim import hudkit, ui  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _display():
    pygame.display.init()
    pygame.display.set_mode((8, 8))
    ui.ALPHA = True
    yield
    ui.ALPHA = False


def test_feed_newest_on_top_repeats_refresh_and_old_ones_go():
    f = hudkit.Feed()
    f.push("bus", "Green held for a bus", "Junction 838", key="838-bus", now=0.0)
    f.push("ambulance", "Ambulance", "Junction 1574", key="1574-amb", now=1.0)
    f.push("bus", "Green held for a bus", "Junction 838", key="838-bus", now=2.0)   # a repeat: no second line
    assert [n.key for n in f.items] == ["1574-amb", "838-bus"] and f.items[1].t > 1.0
    f.push("bus", "Green shortened: a bus is waiting", "Junction 838", key="838-bus", now=2.2)   # news: on top
    assert [n.key for n in f.items] == ["838-bus", "1574-amb"] and len(f.items) == 2
    screen = pygame.Surface((1280, 720), pygame.SRCALPHA)
    rects = f.draw(screen, 1264, 16, now=2.5)
    assert len(rects) == 2 and all(r.right <= 1280 + ui.MARGIN for r in rects)
    assert f.busy(5.0) and not f.busy(20.0) and f.draw(screen, 1264, 16, now=20.0) == []


def test_banner_once_in_a_while():
    b = hudkit.Banner()
    assert b.show("accident", "Junction 1574", "Accident", "2 cars", now=0.0)
    assert not b.show("accident", "Junction 53", "Accident", now=5.0)            # too soon after the last
    screen = pygame.Surface((1280, 720), pygame.SRCALPHA)
    assert b.draw(screen, (1280, 720), now=1.0) and b.busy(1.0)
    assert b.draw(screen, (1280, 720), now=10.0) == [] and not b.busy(10.0)
    assert b.show("accident", "Junction 53", "Accident", now=30.0)


def test_events_become_notices_once():
    f, b = hudkit.Feed(), hudkit.Banner()
    ev = hudkit.Events(f, b)
    loc = NS(x=0.0, y=0.0)
    centres = {"838": (100.0, 0.0, 0), "53": (5000.0, 0.0, 0)}
    old = [{"t": 10.0, "tls": "838", "kind": "bus", "text": "green held for a bus"}]
    ev.update(loc, [], old, centres, [], hudkit.Feed)          # what was there before: no notice
    assert not f.items
    new = old + [{"t": 12.0, "tls": "838", "kind": "bus", "text": "green held for a bus"},
                 {"t": 12.0, "tls": "53", "kind": "bus", "text": "green held for a bus"},        # too far
                 {"t": 12.0, "tls": "838", "kind": "skip", "text": "skipped an empty green"}]   # not a notice kind
    ev.update(loc, [], new, centres, [], None)
    assert [n.kind for n in f.items] == ["bus"] and "838" in f.items[0].line
    m = {"id": 7, "x": 50.0, "y": 0.0, "status": "ACCIDENT", "phase": "crash", "kind": "junction", "label": "severe"}
    ev.update(loc, [m], new, centres, [], lambda m: "2 cars")
    assert b.cur is not None and b.cur[2] == "ACCIDENT"           # near: the banner
    ev.update(loc, [dict(m, status="POLICE ON SCENE", phase="secured")], new, centres, [], lambda m: "2 cars")
    assert f.items[0].kind == "police"
    ev.update(loc, [], new, centres, [], lambda m: "")
    assert f.items[0].kind == "good" and f.items[0].title == "Road cleared"
    ev.update(loc, [], new, centres, [{"t": 30.0, "node": "2014-3", "p": 0.86}], lambda m: "")
    ev.update(loc, [], new, centres, [{"t": 30.0, "node": "2014-3", "p": 0.86}, {"t": 40.0, "node": "53-1", "p": 0.9}],
              lambda m: "", lambda a: "LONG QUEUE LIKELY WITHIN 3 MIN")
    assert f.items[0].kind == "forecast" and "53-1" in f.items[0].line


def test_prompt_chips_and_tags_render():
    p = hudkit.prompt("H   Inspect traffic camera 207-1 looking north      F   Follow car live20")
    c = hudkit.chips([("live", (255, 84, 84)), ("193 vehicles", None)])
    t = hudkit.tag("parking", "Parking Central", "60 free of 60")
    assert p.get_width() > 300 and c.get_height() == 26 and t.get_height() > 44


def test_long_notices_fit_and_the_key_help_reads_its_keys():
    f = hudkit.Feed()
    f.push("ambulance", "Green ended early", "An ambulance is coming  ·  Junction 1574  ·  214 m " * 3, now=0.0)
    card = f._card(f.items[0])
    assert card.get_width() <= hudkit.Feed.MAX_W + 2 * ui.MARGIN
    k = hudkit.keys("F5 pause    F6 / F7  -1 h / +1 h    H inspect a camera    Esc quit")
    assert 400 < k.get_width() < 1000
    ev = hudkit.Events(f, hudkit.Banner())
    ev.update(NS(x=0.0, y=0.0), [], [{"t": 1.0, "tls": "838", "kind": "ambulance", "text": "x"}], {"838": (0, 0, 0)},
              [], None)
    ev.update(NS(x=0.0, y=0.0), [], [{"t": 2.0, "tls": "838", "kind": "ambulance",
                                      "text": "green ended early: an ambulance is coming"}], {"838": (0, 0, 0)}, [], None)
    n = f.items[0]
    assert n.title == "Green ended early" and n.line.startswith("An ambulance is coming")
