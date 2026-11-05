"""GO-FLOW priority agent adapter (cosim/goflow_priority.py): names, the flag, and the child process never being
fatal. Uses a fake child (no second Python environment needed)."""

from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from cosim import goflow_priority as gp
from cosim.controllers import make

FAKE = textwrap.dedent("""
    import json, sys
    for line in sys.stdin:
        m = json.loads(line)
        if m["type"] == "hello":
            print(json.dumps({"type": "ready", "v": 1, "agent": "fake", "version": "0", "config": {}}), flush=True)
        elif m["type"] == "state":
            if m["seq"] == 3:
                sys.exit(1)  # dies on the third state
            print(json.dumps({"type": "plan", "v": 1, "seq": m["seq"], "t": m["t"], "junctions": {"J": {
                "mode": "EMERGENCY_PREEMPTION", "vehicle": "a", "level": 3, "links": [0], "eta_s": 5, "rank": 0}},
                "refused": []}), flush=True)
        elif m["type"] == "bye":
            break
""")


def test_names():
    a = make("agents_goflow")
    assert isinstance(a, gp.GoflowAgents) and a.name == "agents_goflow" and not a.uses_cameras
    b = make("agents_observed_goflow-trickle@ev_patience_s=30")
    assert isinstance(b, gp.ObservedGoflowAgents) and b.uses_cameras
    assert "trickle" not in b.on and b.ev_patience_s == 30.0
    assert type(make("agents")).__name__ == "ActuatedAgents"  # the existing names are unchanged


def _client(tmp_path, mode="lockstep"):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    cl = gp.GoflowClient(mode, timeout_s=5.0, python=sys.executable)
    return cl, fake


def test_client_plan_staleness_and_death(tmp_path, monkeypatch):
    cl, fake = _client(tmp_path)
    monkeypatch.setattr(gp, "AGENT_DIR", tmp_path)
    monkeypatch.setattr(gp.GoflowClient, "_spawn", _spawn_fake(fake))
    assert cl.start({"type": "hello"})
    assert cl.status == "ok"
    cl.exchange({"type": "state", "seq": 1, "t": 10.0, "vehicles": []})
    assert cl.valid_plan(10.5, 2.0)["junctions"]["J"]["links"] == [0]
    assert cl.valid_plan(12.5, 2.0) is None  # older than 2 s of simulation: not used
    cl.exchange({"type": "state", "seq": 2, "t": 11.0, "vehicles": []})
    cl.exchange({"type": "state", "seq": 3, "t": 12.0, "vehicles": []})  # the child dies here
    for _ in range(50):
        if cl.status == "down":
            break
        time.sleep(0.05)
    assert cl.status == "down" and cl.valid_plan(12.0, 2.0) is None
    cl.exchange({"type": "state", "seq": 4, "t": 13.0, "vehicles": []})  # no exception while down
    cl.close()


def test_missing_interpreter_is_not_fatal(tmp_path):
    cl = gp.GoflowClient("live", python=str(tmp_path / "no_python_here"))
    assert cl.start({"type": "hello"}) is False
    assert cl.status == "down"
    cl.exchange({"type": "state", "seq": 1, "t": 0.0, "vehicles": []})
    assert cl.valid_plan(0.0, 2.0) is None


def test_flag(monkeypatch):
    pytest.importorskip("pygame")
    from cosim import freefly

    monkeypatch.delenv("GOFLOW_PRIORITY_MODE", raising=False)
    a = freefly.parse_args(["--controller", "agents_observed", "--goflow-priority"])
    assert a.controller == "agents_observed_goflow"
    import os
    assert os.environ.get("GOFLOW_PRIORITY_MODE") == "live"
    monkeypatch.delenv("GOFLOW_PRIORITY_MODE", raising=False)
    assert freefly.parse_args(["--controller", "actuated", "--goflow-priority"]).controller == "actuated"
    assert freefly.parse_args(["--controller", "agents_observed"]).controller == "agents_observed"
    monkeypatch.delenv("GOFLOW_PRIORITY_ARGS", raising=False)
    freefly.parse_args(["--controller", "agents", "--goflow-priority", "--goflow-llm", "qwen2.5:3b"])
    assert os.environ.get("GOFLOW_PRIORITY_ARGS") == "--llm --llm-model qwen2.5:3b"
    monkeypatch.delenv("GOFLOW_PRIORITY_ARGS", raising=False)
    monkeypatch.delenv("GOFLOW_PRIORITY_MODE", raising=False)


def _spawn_fake(fake):
    import subprocess
    import threading

    def spawn(self):
        self.proc = subprocess.Popen([sys.executable, str(fake)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        self._got.clear()
        if not self._write(self.hello) or not self._got.wait(5) or self.ready is None:
            self._down("no answer")
            return False
        self.status = "ok"
        return True

    return spawn


def test_hello_shape():
    """What the child is told about a junction: its car greens as link sets (agents/priority_agent/town05/serve.py)."""
    msg = {"type": "hello", "v": 1, "junctions": {"J": {"n_links": 8, "greens": [[0, 1], [2, 3]]}}}
    assert json.loads(json.dumps(msg))["junctions"]["J"]["greens"][1] == [2, 3]


def test_request_source_is_off_by_default():
    """The junction agents' public hook: unset on every existing controller, so nothing changes for them."""
    for name in ("agents", "agents_observed", "agents-corridor"):
        a = make(name)
        assert a.request_source is None
        assert not any(k.startswith(("external_", "goflow_")) for k in a.stats)
    g = make("agents_goflow")
    assert hasattr(g, "_gf_source") and not hasattr(gp.GoflowMixin, "_requests")
