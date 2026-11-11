"""The overlay's per-pixel alpha (cosim/overlay.py, cosim/ui.py): the pixels given to Windows, the glass panels."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from cosim import ui  # noqa: E402
from cosim.overlay import premultiplied  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _display():
    pygame.display.init()
    pygame.display.set_mode((8, 8))
    yield
    ui.ALPHA = False


def _random_surface(size=(320, 200)):
    s = pygame.Surface(size, pygame.SRCALPHA)
    rng = np.random.default_rng(3)
    px = pygame.surfarray.pixels3d(s)
    px[:] = rng.integers(0, 256, px.shape, dtype=np.uint8)
    del px
    pa = pygame.surfarray.pixels_alpha(s)
    pa[:] = rng.integers(0, 256, pa.shape, dtype=np.uint8)
    del pa
    return s


@pytest.mark.parametrize("rect", [(0, 0, 320, 200), (17, 9, 120, 60), (200, 150, 120, 50), (3, 5, 33, 7)])
def test_premultiplied_pixels_of_any_rectangle(rect):
    """Each colour times its alpha, in B G R A order, for a rectangle anywhere in the surface: the first version
    was right only for the whole surface (stripes over the picture in CARLA, 11 Nov)."""
    s = _random_surface()
    x, y, w, h = rect
    got = premultiplied(s, rect)
    assert got.shape == (h, w, 4)
    for j, i in ((0, 0), (h - 1, w - 1), (h // 2, w // 3)):
        r, g, b, a = s.get_at((x + i, y + j))
        want = [round(b * a / 255), round(g * a / 255), round(r * a / 255), a]
        assert np.abs(got[j, i].astype(int) - want).max() <= 1
    rgb = pygame.surfarray.array3d(s)[x:x + w, y:y + h].transpose(1, 0, 2).astype(int)
    al = pygame.surfarray.array_alpha(s)[x:x + w, y:y + h].T.astype(int)
    want = (rgb[:, :, ::-1] * al[:, :, None] + 127) // 255
    assert np.abs(got[:, :, :3].astype(int) - want).max() <= 1
    assert (got[:, :, 3] == al).all()


def test_glass_panels_only_with_per_pixel_alpha():
    panel = pygame.Surface((120, 40))
    panel.fill(ui.BG)
    pygame.draw.rect(panel, (255, 80, 80), (0, 0, 5, 40))
    ui.ALPHA = False
    assert ui.glassify(panel) is panel and ui.glass_fast(panel) is panel   # the colour key: as drawn
    ui.ALPHA = True
    for glass in (ui.glassify(panel), ui.glass_fast(panel)):
        assert glass.get_size() == (120 + 2 * ui.MARGIN, 40 + 2 * ui.MARGIN)
        assert glass.get_at((0, 0)).a < 40                                  # the margin: only the shadow
        mid = glass.get_at((ui.MARGIN + 60, ui.MARGIN + 20))
        assert 180 <= mid.a < 255                                           # the body lets the town through
    bar = ui.glassify(panel).get_at((ui.MARGIN + 2, ui.MARGIN + 20))
    assert bar.a == 255 and bar.r > 200                                     # what is drawn on it stays solid
