"""The accident agent's detection process (11 Nov): camera pictures -> Omar's agent (docs/INCIDENT_AGENT_AUDIT.md).

Runs beside the viewer, as the red-light agent (cosim/redlight_proc.py): PyTorch on the GPU here, never in the viewer.
Per camera one YOLO with its own ByteTrack (`track(persist=True)`, as his `process_frames` builds one model per stream),
the CARLA detector with his class map (agents/incident_detection/config/detector_carla_yolo26n_t05.json), his
confidence and image size. Each picture becomes his rows (`rows_of`, the same as his `result_rows`), sent as one
JSON line to his process (agents/incident_detection/serve.py, its own venv), which answers with a verdict per camera
and 5 s window.

in  (q_frames): ("frame", camera, sim s of the picture, JPEG bytes) | ("stop",)
out (q_out):    ("ready", info) | ("window", record) | ("stats", dict) | ("error", text) | ("off", why)

His process: restarted 5, 20 and 60 s after it dies, then left off for the run. Its input is a short queue: when it
falls behind, rows are dropped and counted, YOLO never waits. If this process dies the viewer only loses the agent.
Nothing here reads the simulation: the picture's time is the only number from the town.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "incident_detection"
CONFIG = AGENT / "config" / "detector_carla_yolo26n_t05.json"
WEIGHTS = ROOT / "models" / "carla-yolo26n-t05.pt"
RESTART_S = (5.0, 20.0, 60.0)
TO_AGENT_MAX = 400            # rows waiting for his process (about 20 s of 4 cameras at 5 fps)
STATS_S = 10.0


def agent_python() -> Path:
    return AGENT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def class_ids(class_map: dict, names: dict) -> dict:
    """His DetectorConfig.class_ids: the model's own names -> {class id: his category}; unknown names fail."""
    lookup = {str(n).lower(): int(i) for i, n in dict(names).items()}
    missing = sorted(n for n in class_map if n.lower() not in lookup)
    if missing:
        raise ValueError(f"the detector has no classes {missing}; it knows {sorted(lookup)}")
    return {lookup[n.lower()]: c for n, c in class_map.items()}


def rows_of(result, frame_area: float, categories: dict) -> tuple[dict, list]:
    """His `result_rows` (agents/incident_detection/src/video_processing.py) without the time: one frame row and the
    tracked detections as lists in serve.DET_KEYS order. Boxes without a track id are skipped, as there."""
    confidences, area, dets = [], 0.0, []
    for box in result.boxes or []:
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        confidences.append(confidence)
        area += (x2 - x1) * (y2 - y1)
        dets.append([int(box.id[0].item()), class_id, categories[class_id], confidence, (x1 + x2) / 2, (y1 + y2) / 2,
                     y2 - y1, (x2 - x1) * (y2 - y1)])
    frame = {"vehicle_count": len(confidences), "bbox_coverage": min(area / frame_area, 1.0),
             "mean_detection_confidence": sum(confidences) / len(confidences) if confidences else 0.0}
    return frame, dets


class AgentChild:
    """His process: started, fed through a short queue by a writer thread, read by a reader thread, restarted."""

    def __init__(self, q_out, detector: str, log_path: Path, cmd: list[str] | None = None):
        self.q_out, self.detector, self.log_path = q_out, detector, Path(log_path)
        self.cmd = cmd or [str(agent_python()), "-u", str(AGENT / "serve.py")]
        self.proc = None
        self.inbox: queue.Queue = queue.Queue(maxsize=TO_AGENT_MAX)
        self.restarts, self.next_try, self.off = 0, 0.0, None
        self.stats = {"sent": 0, "dropped": 0, "windows": 0, "errors": 0, "restarts": 0, "ready": False}
        self.bye = threading.Event()
        self.kill_after = float(os.environ.get("GOFLOW_INCIDENT_KILL_AFTER_S", "0") or 0)  # fault test only
        self.t_start = time.time()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        err = open(self.log_path, "a", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                                     encoding="utf-8", bufsize=1, cwd=str(AGENT), creationflags=flags)
        self.inbox = queue.Queue(maxsize=TO_AGENT_MAX)
        threading.Thread(target=self._write, args=(self.proc, self.inbox), daemon=True, name="agent in").start()
        threading.Thread(target=self._read, args=(self.proc,), daemon=True, name="agent out").start()
        self.inbox.put(json.dumps({"op": "hello", "detector": self.detector}))

    def _write(self, proc, inbox) -> None:
        try:
            while True:
                line = inbox.get()
                if line is None:
                    break
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass  # the process died: the next check restarts it

    def _read(self, proc) -> None:
        try:
            for line in proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    self.stats["errors"] += 1
                    continue
                op = msg.pop("op", None)
                if op == "window":
                    self.stats["windows"] += 1
                    self.q_out.put(("window", msg))
                elif op == "ready":
                    self.stats["ready"] = True
                    self.q_out.put(("ready", msg))
                elif op == "error":
                    self.stats["errors"] += 1
                    if self.stats["errors"] <= 20:
                        self.q_out.put(("error", f"agent: {msg.get('text')}"))
                elif op == "bye":
                    self.q_out.put(("bye", msg))
                    self.bye.set()
        except (OSError, ValueError):
            pass

    def send(self, line: str) -> None:
        if self.off is not None:
            return
        try:
            self.inbox.put_nowait(line)
            self.stats["sent"] += 1
        except queue.Full:
            self.stats["dropped"] += 1

    def check(self, now: float) -> None:
        """Restart a dead process (5, 20, 60 s later), then give up for the run."""
        if self.off is not None:
            return
        if self.kill_after and self.alive() and now - self.t_start >= self.kill_after:
            self.kill_after = 0.0
            self.q_out.put(("error", "fault test: the agent's process is killed (GOFLOW_INCIDENT_KILL_AFTER_S)"))
            self.proc.kill()
        if self.alive():
            return
        if self.proc is not None and self.next_try == 0.0:
            if self.restarts >= len(RESTART_S):
                self.off = f"his process died {self.restarts + 1} times"
                self.q_out.put(("off", self.off))
                return
            self.next_try = now + RESTART_S[self.restarts]
            self.q_out.put(("error", f"agent process ended (code {self.proc.poll()}); restart in "
                                     f"{RESTART_S[self.restarts]:.0f} s"))
        if self.next_try and now >= self.next_try:
            self.restarts += 1
            self.stats["restarts"] = self.restarts
            self.next_try = 0.0
            self.stats["ready"] = False
            self.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc is None:
            return
        if self.alive():
            for line in (json.dumps({"op": "flush"}), json.dumps({"op": "stop"}), None):
                try:
                    self.inbox.put(line, timeout=1.0)
                except queue.Full:
                    break
            self.bye.wait(timeout)
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def run(q_frames, q_out, cams: list[str], log_dir: str, weights: str | None = None, config: str | None = None,
        child_cmd: list[str] | None = None, device=None) -> None:
    try:
        _run(q_frames, q_out, cams, Path(log_dir), weights, config, child_cmd, device)
    except Exception:  # noqa: BLE001 - reported to the viewer, which goes on
        q_out.put(("error", traceback.format_exc()))
        q_out.put(("off", "the detection process failed"))


class Detector:
    """The CARLA detector and one ByteTrack per camera. Default (`batch=False`): `track(persist=True)` per picture
    on the camera's own model, his process_frames' way. `batch=True`: the pictures waiting, of all cameras, in one
    detector call, then each camera's tracker in picture order with Ultralytics' own tracker step
    (trackers/track.py on_predict_postprocess_end); tools/incident_agent_detcheck.py compares the two on a clip.
    Beside CARLA on the GTX 1650 neither keeps 4 cameras at 5 fps (97 and 108 ms a picture, 11 Nov)."""

    def __init__(self, weights: str, cfg: dict, cams: list, device, batch: bool = True):
        from ultralytics import YOLO

        self.cfg, self.device, self.batch = cfg, device, batch
        self.half = 16 if device != "cpu" else None
        self.imgsz = int(cfg["imgsz"]) if cfg.get("imgsz") else None
        self.conf = float(cfg.get("confidence", 0.25))
        self.tracker_yaml = cfg.get("tracker", "bytetrack.yaml")
        self.models = {c: YOLO(weights) for c in cams} if not batch else {}
        self.model = YOLO(weights) if batch else None
        names = (self.model or next(iter(self.models.values()))).names
        self.categories = class_ids(cfg["class_map"], names)
        self.trackers = {}
        if batch:
            from ultralytics.trackers.track import TRACKER_MAP
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            tcfg = IterableSimpleNamespace(**YAML.load(check_yaml(self.tracker_yaml)))
            self.trackers = {c: TRACKER_MAP[tcfg.tracker_type](args=tcfg) for c in cams}

    def opts(self) -> dict:
        o = dict(conf=self.conf, classes=sorted(self.categories), device=self.device, verbose=False, save=False,
                 quantize=self.half)
        if self.imgsz:
            o["imgsz"] = self.imgsz
        return o

    def warm_up(self, shape=(540, 960, 3)) -> None:
        import numpy as np

        for m in ([self.model] if self.model is not None else self.models.values()):
            m.predict(np.zeros(shape, np.uint8), **dict(self.opts(), classes=None))  # no box: no tracker learns

    def __call__(self, items: list) -> list:
        """items: [(camera, BGR picture)] in arrival order -> [(frame row, detection rows)] in the same order."""
        if not self.batch:
            out = []
            for cam, bgr in items:
                r = self.models[cam].track(bgr, persist=True, tracker=self.tracker_yaml, **self.opts())[0]
                out.append(rows_of(r, float(bgr.shape[0] * bgr.shape[1]), self.categories))
            return out
        import torch

        results = self.model.predict([bgr for _, bgr in items], **self.opts())
        out = []
        for (cam, bgr), r in zip(items, results):
            tracker = self.trackers[cam]
            tracks = tracker.update(r.boxes.cpu().numpy(), r.orig_img, feats=None)
            if len(tracks) == 0:
                if any(not t.is_activated for t in tracker.tracked_stracks):
                    r = r[:0]
            else:
                idx = tracks[:, -1].astype(int)
                r = r[idx]
                r.update(boxes=torch.as_tensor(tracks[:, :-1], device=r.boxes.data.device))
            out.append(rows_of(r, float(bgr.shape[0] * bgr.shape[1]), self.categories))
        return out


def _run(q_frames, q_out, cams, log_dir: Path, weights, config, child_cmd, device) -> None:
    import cv2
    import numpy as np
    import torch

    cfg = json.loads(Path(config or CONFIG).read_text(encoding="utf-8"))
    weights = str(weights or os.environ.get("GOFLOW_YOLO_WEIGHTS") or WEIGHTS)
    if device is None:
        if not torch.cuda.is_available():
            q_out.put(("off", "no CUDA: the accident agent needs the GPU (CPU YOLO cannot keep 4 cameras at 5 fps)"))
            return
        device = 0
    # one track() per picture, his way, by default: batching was no faster live (108 against 97 ms a picture beside
    # CARLA, 11 Nov: the GPU is CARLA's); GOFLOW_INCIDENT_BATCH=1 for the batched path
    batch = os.environ.get("GOFLOW_INCIDENT_BATCH", "0") == "1"
    det = Detector(weights, cfg, list(cams), device, batch=batch)
    det.warm_up()
    child = AgentChild(q_out, cfg["name"], log_dir / "agent_stderr.log", child_cmd)
    child.start()
    q_out.put(("ready", {"detector": cfg["name"], "weights": Path(weights).name, "device": str(device),
                         "cameras": list(cams), "classes": {int(k): v for k, v in det.categories.items()},
                         "conf": det.conf, "imgsz": det.imgsz, "fp16": det.half is not None, "batch": batch}))
    took, sizes, pictures, bad, t_stats, stop = [], [], 0, 0, time.time(), False
    while not stop:
        now = time.time()
        child.check(now)
        if now - t_stats >= STATS_S:
            ms, bs = sorted(took), sorted(sizes)
            q_out.put(("stats", dict(child.stats, pictures=pictures, bad_pictures=bad, off=child.off,
                                     detect_ms_per_picture_median=round(ms[len(ms) // 2], 1) if ms else None,
                                     detect_ms_per_picture_p95=round(ms[int(len(ms) * 0.95)], 1) if ms else None,
                                     batch_median=bs[len(bs) // 2] if bs else None)))
            took, sizes, t_stats = [], [], now
        try:
            items = [q_frames.get(timeout=0.5)]
        except queue.Empty:
            continue
        while len(items) < (2 * len(cams) if batch else 1):  # whatever else is waiting, into the same call
            try:
                items.append(q_frames.get_nowait())
            except queue.Empty:
                break
        work = []
        for item in items:
            if item[0] == "stop":
                stop = True
                continue
            _, cam, sim_t, jpeg = item
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None or cam not in cams:
                bad += 1
                continue
            work.append((cam, sim_t, bgr))
        if not work:
            continue
        t1 = time.perf_counter()
        rows = det([(c, b) for c, _, b in work])
        took.append((time.perf_counter() - t1) * 1000.0 / len(work))
        sizes.append(len(work))
        for (cam, sim_t, _), (frame, dets) in zip(work, rows):
            pictures += 1
            child.send(json.dumps({"op": "frame", "cam": cam, "t": round(float(sim_t), 3), "frame": frame,
                                   "dets": dets}, separators=(",", ":")))
    child.stop()
    q_out.put(("stats", dict(child.stats, pictures=pictures, bad_pictures=bad, off=child.off, final=True)))
