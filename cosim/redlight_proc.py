"""The red-light agent in a process of its own, beside the viewer (31 Oct).

As the camera inspector's detector (cosim/inspector.py): loading PyTorch in the viewer's own process would hold its
picture up, so the agent (cosim/redlight.SiteAgent: the detector on the GPU, the tracker, the rule, the plate reader,
the evidence) runs here. Messages in (q_frames):
  ("frame", picture, SUMO time, JPEG bytes)   the site camera's pictures (cosim/enforcement.SiteCamera)
and (q_plates, optional):
  ("plate", picture, SUMO time, JPEG bytes)   the plate camera's pictures (kept a few seconds, read when needed)
and (q_in):
  ("signal", SUMO time, state)                the junction controller's log, one line per SUMO step
  ("clock", SUMO time, town seconds, speed, weekday)   to print the town's date and time on the evidence
  ("stop",)
out: ("ready", detector) once loaded, ("case", summary) for each case written to out_dir, ("error", text).
Pictures that come while it is busy are dropped by the viewer (its queue is short), as a real camera's processor
would skip frames rather than fall behind.
"""

from __future__ import annotations

import queue
import traceback


def run(q_in, q_frames, q_out, site_json: dict, model: str, out_dir: str, q_plates=None) -> None:
    """q_in: the control messages (signal, clock, stop); q_frames: the pictures (a short queue: the viewer drops
    pictures rather than wait); q_plates: the plate camera's pictures."""
    try:
        _run(q_in, q_frames, q_out, site_json, model, out_dir, q_plates)
    except Exception:  # noqa: BLE001 - reported to the viewer, which goes on
        q_out.put(("error", traceback.format_exc()))


def _run(q_in, q_frames, q_out, site_json: dict, model: str, out_dir: str, q_plates=None) -> None:
    import cv2
    import numpy as np

    from cosim import enforcement as enf
    from cosim import redlight

    site = enf.Site.from_json(site_json)
    log_ = enf.SignalLog()
    clock_ref = {"t": None}

    def clock(t: float) -> str:
        c = clock_ref
        if c["t"] is None:
            return f"t = {t:.2f} s"
        s = (c["game"] + (t - c["t"]) * c["speed"]) % 86400
        return f"{c['day']} {int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"

    try:
        det = redlight.UltralyticsDetector(model)
        name = "ultralytics"
    except Exception:  # noqa: BLE001 - no PyTorch here: the ONNX export if there is one
        from pathlib import Path

        onnx = Path(model).with_suffix(".onnx")
        alt = Path(__file__).resolve().parents[1] / "out" / "detector_onnx" / (Path(model).stem + ".onnx")
        path = onnx if onnx.is_file() else alt
        det = redlight.OnnxDetector(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        name = "onnxruntime"
    agent = redlight.SiteAgent(site, log_, det, out_dir, clock=clock)
    q_out.put(("ready", name))
    while True:
        stop = False
        while True:  # the controller's log and the clock first: a picture is judged with the log up to its time
            try:
                msg = q_in.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "stop":
                stop = True
            elif msg[0] == "signal":
                log_.add(msg[1], msg[2])
            elif msg[0] == "clock":
                clock_ref.update(t=msg[1], game=msg[2], speed=msg[3], day=msg[4])
        if stop:
            break
        while q_plates is not None:     # the plate camera's pictures: kept as they are (JPEG) until one is needed
            try:
                _, kp, tp, jp = q_plates.get_nowait()
            except queue.Empty:
                break
            agent.plate_picture(kp, tp, jp)
        try:
            _, k, t, jpeg = q_frames.get(timeout=0.05)
        except queue.Empty:
            continue
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        for rec in agent.frame(k, t, bgr[:, :, ::-1].copy()):
            c = rec["case"]
            q_out.put(("case", {"id": c["id"], "plate": c["plate"], "conf": c["plate_conf"], "red_s": c["red_s"],
                                "when": rec["when"], "dir": str(agent.writer.out / c["id"])}))
