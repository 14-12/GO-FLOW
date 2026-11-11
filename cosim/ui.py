"""Shared look of the overlay's panels (11 Nov): glass with rounded corners, a hairline border and a soft shadow.

The overlay window used to be see-through by a colour key: a pixel was solid or absent, so panels were square,
solid, and text sat on its own solid background. With per-pixel alpha (cosim/overlay.AlphaPresenter) a panel can
let the town show through and have smooth edges. `glassify` turns a panel drawn the old way (solid BG, text on BG)
into that, so the panels keep their own drawing code. With the colour key (ALPHA False) it returns the panel as is.
"""

from __future__ import annotations

import numpy as np
import pygame

ALPHA = False       # set by the viewer once the overlay window takes per-pixel alpha
BG = (22, 24, 30)   # the panels' background (cosim/agent_hud.BG)
MARGIN = 10         # room round a panel for its shadow
_cache: dict = {}


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Alpha (w, h) of a rounded rectangle with smooth corners (drawn 4x, scaled down)."""
    key = ("mask", w, h, radius)
    m = _cache.get(key)
    if m is None:
        big = pygame.Surface((w * 4, h * 4), pygame.SRCALPHA)
        pygame.draw.rect(big, (255, 255, 255, 255), big.get_rect(), border_radius=radius * 4)
        m = _cache[key] = pygame.surfarray.array_alpha(pygame.transform.smoothscale(big, (w, h)))
    return m


def _shadow(w: int, h: int, radius: int, strength: int) -> pygame.Surface:
    """A soft shadow for a w x h panel, on a surface MARGIN larger on every side."""
    key = ("shadow", w, h, radius, strength)
    s = _cache.get(key)
    if s is None:
        W, H = w + 2 * MARGIN, h + 2 * MARGIN
        base = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.rect(base, (0, 0, 0, strength), (MARGIN - 2, MARGIN, w + 4, h + 3), border_radius=radius + 2)
        small = pygame.transform.smoothscale(base, (max(1, W // 5), max(1, H // 5)))
        s = _cache[key] = pygame.transform.smoothscale(small, (W, H))
    return s


def _border(w: int, h: int, radius: int, alpha: int) -> pygame.Surface:
    key = ("border", w, h, radius, alpha)
    s = _cache.get(key)
    if s is None:
        big = pygame.Surface((w * 3, h * 3), pygame.SRCALPHA)
        pygame.draw.rect(big, (255, 255, 255, alpha), big.get_rect(), width=3, border_radius=radius * 3)
        s = _cache[key] = pygame.transform.smoothscale(big, (w, h))
    return s


def _trim() -> None:
    if len(_cache) > 400:
        _cache.clear()


def glass_fast(surf: pygame.Surface, radius: int = 12, body_alpha: int = 228, shadow: int = 120, border: int = 36
               ) -> pygame.Surface:
    """As glassify for a panel redrawn every picture (the mini-map): the whole panel at one transparency, with
    pygame's own blits (0.2 ms where glassify takes 10). Unchanged without per-pixel alpha."""
    if not ALPHA:
        return surf
    _trim()
    w, h = surf.get_size()
    key = ("fastmask", w, h, radius, body_alpha)
    mask = _cache.get(key)
    if mask is None:
        mask = pygame.Surface((w, h), pygame.SRCALPHA)
        mask.fill((255, 255, 255, 255))
        ma = pygame.surfarray.pixels_alpha(mask)
        ma[:] = (_rounded_mask(w, h, radius).astype(np.uint16) * body_alpha // 255).astype(np.uint8)
        del ma
        _cache[key] = mask
    out = pygame.Surface((w + 2 * MARGIN, h + 2 * MARGIN), pygame.SRCALPHA)
    panel = out.subsurface((MARGIN, MARGIN, w, h))
    panel.blit(surf, (0, 0))                                   # solid, onto nothing: copied as it is
    panel.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    if shadow:   # under the panel: only where the panel leaves room (its margin and what shows through it)
        under = _shadow(w, h, radius, shadow).copy()
        under.blit(out, (0, 0))
        out = under
    if border:
        out.blit(_border(w, h, radius, border), (MARGIN, MARGIN))
    return out


def glassify(surf: pygame.Surface, radius: int = 8, body_alpha: int = 214, shadow: int = 120, border: int = 30,
             bg=BG) -> pygame.Surface:
    """A solid panel (background `bg`, anything on it) as glass: the background lets the town through, what is
    drawn on it stays solid; rounded corners, a hairline border, a soft shadow. The result is MARGIN larger on
    every side (the shadow). Unchanged when the overlay has no per-pixel alpha."""
    if not ALPHA:
        return surf
    _trim()
    w, h = surf.get_size()
    radius = min(radius, h // 2, w // 2)
    rgb = pygame.surfarray.array3d(surf).astype(np.int16)
    diff = np.abs(rgb - np.array(bg, dtype=np.int16)).max(axis=2)
    a = np.clip(body_alpha + diff * 4, 0, 255).astype(np.uint16)
    a = (a * _rounded_mask(w, h, radius).astype(np.uint16) // 255).astype(np.uint8)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.blit(surf, (0, 0))
    pa = pygame.surfarray.pixels_alpha(panel)
    pa[:] = a
    del pa
    if border:
        panel.blit(_border(w, h, radius, border), (0, 0))
        pa = pygame.surfarray.pixels_alpha(panel)  # the border must not put back the cut corners
        np.minimum(pa, _rounded_mask(w, h, radius), out=pa)
        del pa
    out = pygame.Surface((w + 2 * MARGIN, h + 2 * MARGIN), pygame.SRCALPHA)
    if shadow:
        out.blit(_shadow(w, h, radius, shadow), (0, 0))
    out.blit(panel, (MARGIN, MARGIN))
    return out
