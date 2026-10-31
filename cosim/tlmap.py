"""Which SUMO signal links each CARLA traffic light shows.

A CARLA traffic light is created from an OpenDRIVE signal (a landmark of type 1000001). Two
ways to know which SUMO links (tls id, link index) a landmark stands for:

* Maps this project derives from a SUMO network (opendrive.py): netconvert writes one signal
  per controlled link and names it ``<tlsID>_<linkIndex>``, so the id says it.
* Maps that start as OpenDRIVE (CARLA's towns): CARLA's netconvert_carla.py builds the SUMO
  network and records, for every link of every tlLogic, the landmark that controls it as a
  ``<param key="linkSignalID:<linkIndex>" value="<landmark id>"/>``.

Both give ``{landmark id: [(tls id, link index), ...]}``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

Links = dict[str, list[tuple[str, int]]]


def links_from_landmark_ids(landmark_ids: Iterable[str], tls_ids: Iterable[str]) -> Links:
    """``<tlsID>_<linkIndex>`` ids. tls ids may themselves contain ``_``, so the split is
    accepted only when the prefix is a real tls id of the network."""
    known = set(tls_ids)
    out: Links = {}
    for lid in landmark_ids:
        tls, sep, idx = lid.rpartition("_")
        if sep and tls in known and idx.isdigit():
            out.setdefault(lid, []).append((tls, int(idx)))
    return out


def links_from_net(net_file: Path) -> Links:
    """``linkSignalID`` parameters of the tlLogics in a SUMO network."""
    out: Links = {}
    seen: set[tuple[str, int]] = set()
    for _, el in ET.iterparse(str(net_file), events=("end",)):
        if el.tag != "tlLogic":
            continue
        tls = el.get("id")
        for p in el.iter("param"):
            key = p.get("key", "")
            if key.startswith("linkSignalID:"):
                link = int(key.split(":", 1)[1])
                if (tls, link) not in seen:  # one tlLogic per program: count each link once
                    seen.add((tls, link))
                    out.setdefault(p.get("value"), []).append((tls, link))
        el.clear()
    return out
