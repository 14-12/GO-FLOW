"""Which links of a Town05 signal can be green together, read from the program the town sends.

On Town05 a priority request is a (junction, link index) pair, the link the vehicle will use
(TraCI ``vehicle.getNextTLS``), not an approach N/E/S/W: Town05's junctions are skewed, some have
3 legs, and their approaches have 1 to 6 lanes, so the N/E/S/W model of ``environment.intersection``
does not hold there. Two requests at one junction are compatible when one car green serves both.
"""
from __future__ import annotations


class LinkMap:
    def __init__(self, junctions: dict[str, dict]):
        """junctions: the ``hello`` message's {tls: {"n_links": int, "greens": [[link, ...], ...]}}
        (car greens only, program order)."""
        self.greens: dict[str, list[frozenset[int]]] = {
            tls: [frozenset(int(x) for x in g) for g in j.get("greens", [])] for tls, j in junctions.items()}
        self.n_links = {tls: int(j.get("n_links", 0)) for tls, j in junctions.items()}

    def serving(self, tls: str, link: int) -> list[int]:
        """Indices (in the car greens) of the greens that give this link its green."""
        return [k for k, g in enumerate(self.greens.get(tls, [])) if int(link) in g]

    def compatible(self, tls: str, a: int, b: int) -> bool:
        return any(int(a) in g and int(b) in g for g in self.greens.get(tls, []))

    def known(self, tls: str, link: int) -> bool:
        return bool(self.serving(tls, link))


class TownJunction:
    """What ``coordination.rules`` reads from an ``Intersection``, for a Town05 junction:
    ``incoming`` is keyed by link index (as a string), the 'approach' of a Town05 stop."""

    def __init__(self, tls: str, links: LinkMap):
        self.tls_id = tls
        n = links.n_links.get(tls, 0)
        self.incoming = {str(i): str(i) for i in range(n) if links.known(tls, i)}
        self.approach_phase = {a: (links.serving(tls, int(a)) or [0])[0] for a in self.incoming}
