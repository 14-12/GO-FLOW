"""Construit les réseaux SUMO du projet.

- single   : un carrefour isolé (C) à 4 branches de 400 m  -> entraînement PPO
- corridor : artère Est-Ouest avec 3 carrefours à feux (J1, J2, J3) espacés de 300 m,
             chaque carrefour ayant une rue transversale Nord-Sud de 250 m -> évaluation

Pour chaque réseau, deux variantes de feux :
- <name>.net.xml           : feux à cycle fixe (static), 90 s
- <name>_actuated.net.xml  : feux actionnés par détecteurs (baseline B2)

Usage : python simulation/build_networks.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from tools.sumo_interface import sumo_tool

NET_DIR = Path(__file__).parent / "network"
SPEED = 13.89  # 50 km/h
LANES = 2


def netconvert() -> str:
    return sumo_tool("netconvert")


def _write(name: str, nodes: list[tuple], edges: list[tuple[str, str]]) -> None:
    nod = ["<nodes>"] + [
        f'  <node id="{i}" x="{x}" y="{y}" type="{t}"/>' for i, x, y, t in nodes
    ] + ["</nodes>"]
    edg = ["<edges>"] + [
        f'  <edge id="{a}2{b}" from="{a}" to="{b}" numLanes="{LANES}" speed="{SPEED}"/>'
        for a, b in edges
    ] + ["</edges>"]
    (NET_DIR / f"{name}.nod.xml").write_text("\n".join(nod) + "\n")
    (NET_DIR / f"{name}.edg.xml").write_text("\n".join(edg) + "\n")


def _convert(name: str) -> None:
    base = [netconvert(), "-n", str(NET_DIR / f"{name}.nod.xml"), "-e", str(NET_DIR / f"{name}.edg.xml"),
            "--tls.layout", "opposites", "--tls.left-green.time", "0", "--no-turnarounds", "true",
            "--tls.yellow.time", "3", "--tls.cycle.time", "90", "--no-warnings", "true"]
    subprocess.run(base + ["-o", str(NET_DIR / f"{name}.net.xml")], check=True)
    subprocess.run(base + ["--tls.default-type", "actuated",
                           "-o", str(NET_DIR / f"{name}_actuated.net.xml")], check=True)


def build_single() -> None:
    nodes = [("C", 0, 0, "traffic_light"), ("N", 0, 400, "priority"), ("S", 0, -400, "priority"),
             ("E", 400, 0, "priority"), ("W", -400, 0, "priority")]
    edges = []
    for o in "NSEW":
        edges += [(o, "C"), ("C", o)]
    _write("single", nodes, edges)
    _convert("single")


def build_corridor() -> None:
    xs = {"J1": 0, "J2": 300, "J3": 600}
    nodes = [("W0", -300, 0, "priority"), ("E0", 900, 0, "priority")]
    edges = [("W0", "J1"), ("J1", "W0"), ("J1", "J2"), ("J2", "J1"), ("J2", "J3"), ("J3", "J2"),
             ("J3", "E0"), ("E0", "J3")]
    for j, x in xs.items():
        k = j[1]
        nodes += [(j, x, 0, "traffic_light"), (f"N{k}", x, 250, "priority"), (f"S{k}", x, -250, "priority")]
        edges += [(f"N{k}", j), (j, f"N{k}"), (f"S{k}", j), (j, f"S{k}")]
    _write("corridor", nodes, edges)
    _convert("corridor")


if __name__ == "__main__":
    build_single()
    build_corridor()
    print("Réseaux générés dans", NET_DIR)
