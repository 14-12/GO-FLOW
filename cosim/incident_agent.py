"""Omar's accident agent beside the viewer, in shadow (11 Nov; docs/INCIDENT_AGENT_AUDIT.md).

`python -m cosim.freefly --incident-agent "2014-2 looking W;2014-3 looking E"` puts a CARLA camera at each named
camera's pose (out/town05_cameras/layout.json, 960x540, 75 deg), takes 5 pictures a second (his agent's rate), and
hands each one, as JPEG on a short queue, to the detection process (cosim/incident_agent_proc.py), which drops
pictures rather than fall behind. Back come his verdicts, one per camera and 5 s window.

Shadow: a verdict is shown (a note when the crash-moment channel fires) and written to
out/incident_agent/<time>/windows.jsonl. It is never a report: the police, the closures and the route alerts still
learn of an accident only from cosim/incidents.py's call (police_record "sent"). His own rule and ours: nothing acts
until a model meets the agreed criteria (at most 1 false alarm per camera-hour, 80 % of clear crashes, median 30 s,
all within 120 s) on runs it never saw.

Each picture's time is the SUMO second it shows (FreeFly.depicted_at through CARLA's own time of the picture, as
cosim/clip_recorder.ClipRecorder times its pictures), so a verdict lines up with the simulation's record, which
grades it afterwards (tools/eval_incident_agent.py). Nothing here reads that record.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("incident_agent")

FPS = 5.0                  # his agent's rate (TARGET_FPS); his crash moment assumes it (crash_features.FPS)
JPEG_QUALITY = 92          # as the red-light agent's pictures
RAW_MAX = 8                # pictures waiting for the JPEG thread
FRAMES_MAX = 8             # JPEGs waiting for the detection process (the rest are dropped)
NOTE_S = 6.0


class IncidentAgent:
    def __init__(self, world, cams: dict, width: int, height: int, fov: float, sim_now, depicted=None,
                 out_dir: Path | None = None, fps: float = FPS, process: bool = True):
        """cams: camera name -> cosim.perception.CamSpec; sim_now: SUMO's newest second; depicted: wall time -> the
        SUMO second a picture taken then shows (FreeFly.depicted_at)."""
        self.world, self.cams, self.w, self.h, self.fov = world, dict(cams), width, height, fov
        self.sim_now, self.depicted, self.fps = sim_now, depicted, fps
        self.out = Path(out_dir or ROOT / "out" / "incident_agent" / time.strftime("%Y%m%d_%H%M%S"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.sensors, self.offset, self._tick_id = [], None, None
        self.raw: queue.Queue = queue.Queue(maxsize=RAW_MAX)
        self.proc = self.q_frames = self.q_out = None
        self.ready, self.off, self.info = False, None, {}
        self.latest: dict[str, dict] = {}
        self.on = {c: {"v1": False, "impact": False} for c in self.cams}
        self.counts = {"pictures": 0, "raw_dropped": 0, "sent": 0, "dropped": 0, "windows": 0, "v1_alarm_windows": 0,
                       "impact_alarm_windows": 0, "v1_episodes": 0, "impact_episodes": 0, "errors": 0}
        self.lag_s: list[float] = []      # sim seconds from a window's end to its verdict reaching the viewer
        self.proc_stats: dict = {}
        self._windows = open(self.out / "windows.jsonl", "a", encoding="utf-8")
        self._stop = threading.Event()
        self._enc = None
        self.started = None
        if process:
            self._start_process()

    # -- life -------------------------------------------------------------------------------------

    def _start_process(self) -> None:
        import multiprocessing as mp

        from cosim import incident_agent_proc

        ctx = mp.get_context("spawn")
        self.q_frames, self.q_out = ctx.Queue(maxsize=FRAMES_MAX), ctx.Queue()
        self.proc = ctx.Process(target=incident_agent_proc.run, args=(self.q_frames, self.q_out, list(self.cams),
                                                                      str(self.out)), daemon=True,
                                name="incident agent")
        self.proc.start()

    def start_cameras(self) -> None:
        """The CARLA cameras (call once the world is up) and the JPEG thread."""
        import carla

        if self.depicted is not None:
            def on_tick(snap):
                off = time.time() - snap.timestamp.elapsed_seconds
                self.offset = off if self.offset is None else min(self.offset + 1e-4, off)

            self._tick_id = self.world.on_tick(on_tick)
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", str(self.w)), ("image_size_y", str(self.h)), ("fov", str(self.fov)),
                     ("sensor_tick", f"{1.0 / self.fps:.4f}"), ("motion_blur_intensity", "0.0")):
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        for name, spec in self.cams.items():
            tf = carla.Transform(carla.Location(spec.x, spec.y, spec.z), carla.Rotation(spec.pitch, spec.yaw, 0.0))
            s = self.world.spawn_actor(bp, tf)
            s.listen(lambda img, name=name: self._on_image(name, img))
            self.sensors.append(s)
        self._enc = threading.Thread(target=self._encode, daemon=True, name="incident agent jpeg")
        self._enc.start()
        self.started = self.sim_now()
        self._write_run()
        log.info("accident agent (shadow): %d cameras at %.0f fps -> %s", len(self.cams), self.fps, self.out)

    def picture_sim(self, carla_s: float, wall: float, sumo_s: float) -> float:
        """The SUMO second a picture shows (cosim/clip_recorder.ClipRecorder.picture_sim)."""
        if self.depicted is None:
            return sumo_s
        sim = self.depicted(carla_s + self.offset if self.offset is not None else wall)
        return sumo_s if sim is None else sim

    def _on_image(self, name: str, img) -> None:
        if self._stop.is_set():
            return
        wall, sumo_s = time.time(), self.sim_now()
        try:  # timed here (depicted_at keeps the last steps only); encoded in the JPEG thread
            self.raw.put_nowait((name, bytes(img.raw_data), self.picture_sim(img.timestamp, wall, sumo_s)))
            self.counts["pictures"] += 1
        except queue.Full:
            self.counts["raw_dropped"] += 1

    def _encode(self) -> None:
        import cv2
        import numpy as np

        while not self._stop.is_set():
            try:
                name, raw, sim = self.raw.get(timeout=0.5)
            except queue.Empty:
                continue
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3]
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                self.send(name, sim, buf.tobytes())

    def send(self, name: str, sim: float, jpeg: bytes) -> bool:
        if self.q_frames is None or self.off is not None:
            return False
        try:
            self.q_frames.put_nowait(("frame", name, sim, jpeg))
            self.counts["sent"] += 1
            return True
        except queue.Full:  # the detection process is busy: this picture is skipped, as a camera's processor would
            self.counts["dropped"] += 1
            return False

    # -- verdicts ---------------------------------------------------------------------------------

    def poll(self) -> list[str]:
        """His messages since the last call; the notes to show (a crash moment seen)."""
        notes = []
        if self.q_out is None:
            return notes
        while True:
            try:
                kind, body = self.q_out.get_nowait()
            except queue.Empty:
                break
            except Exception:  # noqa: BLE001 - the queue's process is gone
                break
            if kind == "window":
                notes += self.verdict(body)
            elif kind == "ready":
                self.info.update(body)
                if "model" in body:
                    self.ready = True
                    log.info("accident agent ready: model %s (%s), detector %s", body["model"].get("version"),
                             body["model"].get("sha256_12"), body.get("detector"))
                else:
                    log.info("accident agent detection process ready: %s", body)
                self._write_run()
            elif kind == "stats":
                self.proc_stats = body
            elif kind == "error":
                self.counts["errors"] += 1
                log.warning("accident agent: %s", body)
            elif kind == "off":
                self.off = body
                log.error("accident agent off: %s", body)
        if self.proc is not None and not self.proc.is_alive() and self.off is None and not self._stop.is_set():
            self.off = f"the detection process ended (code {self.proc.exitcode})"
            log.error("accident agent off: %s", self.off)
        return notes

    def verdict(self, w: dict) -> list[str]:
        now, wall = self.sim_now(), time.time()
        shown = self.depicted(wall) if self.depicted is not None else None
        w["received_sim_s"] = round(now, 2)       # SUMO's newest second when the verdict reached the viewer
        w["received_shown_s"] = None if shown is None else round(shown, 2)  # ... and the second the pictures show then
        w["received_wall"] = round(wall, 3)
        self._windows.write(json.dumps(w, separators=(",", ":")) + "\n")
        self._windows.flush()
        cam = w.get("camera")
        self.latest[cam] = w
        self.counts["windows"] += 1
        # the verdict's delay on the pictures' own clock (the picture shows SUMO a second or two late, as the viewer)
        self.lag_s.append((now if shown is None else shown) - float(w["sim_end_s"]))
        notes = []
        st = self.on.setdefault(cam, {"v1": False, "impact": False})
        for ch, alarm in (("v1", bool(w.get("incident_detected"))), ("impact", bool(w.get("impact_alarm")))):
            if alarm:
                self.counts[f"{ch}_alarm_windows"] += 1
                if not st[ch]:
                    self.counts[f"{ch}_episodes"] += 1
                    log.info("accident agent: %s alarm on %s (window %.0f-%.0f s, shadow)", ch, cam, w["sim_start_s"],
                             w["sim_end_s"])
                    if ch == "impact":
                        notes.append(f"CAMERA {cam}: crash moment seen (accident agent, shadow: not reported)")
            st[ch] = alarm
        return notes

    def status(self) -> str:
        if self.off is not None:
            return f"accident agent off: {self.off}"
        if not self.ready:
            return "accident agent: starting"
        v1 = sum(s["v1"] for s in self.on.values())
        imp = sum(s["impact"] for s in self.on.values())
        ms = self.proc_stats.get("detect_ms_per_picture_median")
        return (f"accident agent (shadow): {len(self.cams)} cameras, v1 alarm on {v1}, crash moment {imp}"
                + (f", detector {ms:.0f} ms a picture" if ms else ""))

    def summary(self) -> dict:
        lag = sorted(self.lag_s)
        sent = self.counts["sent"] + self.counts["dropped"]
        return {"out": str(self.out), "cameras": list(self.cams), "fps": self.fps, "ready": self.ready, "off": self.off,
                "info": self.info, "counts": dict(self.counts), "process": self.proc_stats,
                "dropped_share": round(self.counts["dropped"] / sent, 4) if sent else None,
                "verdict_lag_s": {"median": round(lag[len(lag) // 2], 2), "p95": round(lag[int(len(lag) * 0.95)], 2),
                                  "max": round(lag[-1], 2)} if lag else None}

    def _write_run(self) -> None:
        cams = {n: (dict(vars(s)) if hasattr(s, "__dict__") else str(s)) for n, s in self.cams.items()}
        run = {"cameras": cams, "width": self.w, "height": self.h,
               "fov": self.fov, "fps": self.fps, "started_sim_s": self.started, "info": self.info,
               "shadow": True, "acts": False}
        (self.out / "run.json").write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")

    def close(self, timeout: float = 15.0) -> dict:
        self._stop.set()
        for s in self.sensors:
            try:
                s.stop()
                s.destroy()
            except Exception as e:  # noqa: BLE001
                log.debug("accident agent camera: %s", e)
        self.sensors = []
        if self._tick_id is not None:
            try:
                self.world.remove_on_tick(self._tick_id)
            except Exception:  # noqa: BLE001
                pass
            self._tick_id = None
        if self.proc is not None:
            try:
                self.q_frames.put(("stop",), timeout=2.0)
            except Exception:  # noqa: BLE001 - full or gone
                pass
            t_end = time.time() + timeout
            while self.proc.is_alive() and time.time() < t_end:
                self.poll()
                time.sleep(0.2)
            self.poll()
            if self.proc.is_alive():
                self.proc.terminate()
        self._windows.close()
        out = self.summary()
        (self.out / "summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        return out
