"""OrbbecRigNode — every Orbbec camera of the rig in ONE process.

The Orbbec SDK does not share GMSL cameras across processes: with one
CameraNode per camera, each process enumerating the bus saw short or wrong
device lists ("Property not found to register callback", a bogus serial, or a
listed camera that never opens) while the others claimed theirs. In one process
with one ``pyorbbecsdk.Context`` all four open and stream concurrently — the
way recording_studio drives them.

One reader thread per camera keeps the latest frame; :meth:`step` publishes each
new frame as ``{name}/{camera}`` (e.g. ``cams/top``) in exactly CameraNode's
payload shape (``images``, ``timestamp``, ``depth_data``, ``intrinsics``,
``extrinsics``), so consumers are unchanged. With ``stereo`` the cameras stream
their left and right colour sensors (no on-board depth): ``{name}/{camera}``
keeps the left image as ``rgb`` and takes its ``depth_data`` from the latest
``{depth_topic}/{camera}`` a stereo matcher publishes (r2s2rn's
Fast-FoundationStereo server), with that depth's source time as
``depth_timestamp``; the pair itself goes out as ``{name}/stereo_{camera}`` at
``stereo_freq`` with the calibration, for the matcher only. Recording uses one
:class:`~robots_realtime.runtime.recording.RgbdWriter` per camera, named
``{name}_{camera}``, so the files are the same as with one node per camera.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from robots_realtime.runtime.node import Node, NodeRole


class OrbbecRigNode(Node):
    """Publish all rig cameras from one process.

    Args:
        name: Node name (topics are ``{name}/{camera}``).
        cameras: ``[{camera: top, serial: CV3H46000036}, ...]``.
        width/height/fps/enable_depth/work_mode: Passed to every
            :class:`~robots_realtime.sensors.cameras.orbbec_camera.OrbbecCamera`.
        poll_freq: Publish loop rate [Hz].
        publish_freq: Bus rate cap per camera [Hz]; recording still gets every frame.
        record: Record RGB-D when the session records.
        stereo: Stream each camera's colour pair instead of colour + depth.
        stereo_freq: Rate of the ``stereo_{camera}`` pairs [Hz].
        depth_topic: Node whose ``{depth_topic}/{camera}`` depth fills ``depth_data`` in stereo.
    """

    role = NodeRole.SENSOR
    published_topics: list[str] = []

    def __init__(
        self,
        name: str = "cams",
        cameras: list[dict] | None = None,
        width: int = 848,
        height: int = 530,
        fps: int = 15,
        enable_depth: bool = True,
        work_mode: str = "Default",
        rig_yaml: str | None = None,
        poll_freq: float = 30.0,
        publish_freq: float | None = None,
        record: bool = True,
        stereo: bool = False,
        stereo_freq: float = 2.0,
        depth_topic: str = "ffs",
        writer=None,
        **kwargs: Any,
    ) -> None:
        self.poll_freq = float(poll_freq)
        if publish_freq is not None:
            self.publish_freq = float(publish_freq)
        super().__init__(name=name, writer=None, **kwargs)
        self._specs = list(cameras or [])
        self.published_topics = [str(c["camera"]) for c in self._specs]
        self._stereo = bool(stereo)
        self._stereo_period = 1.0 / float(stereo_freq) if stereo_freq else 0.0
        self._depth_topic = str(depth_topic)
        self._stereo_sent: dict[str, float] = {}
        if self._stereo:
            from robots_realtime.sensors.cameras.orbbec_camera import DUAL_COLOR_MODE

            work_mode = DUAL_COLOR_MODE
            self.subscribed_topics = [f"{self._depth_topic}/{c['camera']}" for c in self._specs]
        self._cam_kwargs = dict(width=width, height=height, fps=fps, enable_depth=enable_depth,
                                work_mode=work_mode, rig_yaml=rig_yaml, stereo=self._stereo)
        self._fps = float(fps)
        self._record = bool(record)
        self._cams: dict[str, Any] = {}
        self._latest: dict[str, Any] = {}
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._writers: dict[str, Any] = {}
        self._ctx = None

    def setup(self) -> None:
        import pyorbbecsdk as ob

        from robots_realtime.sensors.cameras.orbbec_camera import OrbbecCamera, prepare

        prepare([str(c["serial"]) for c in self._specs], self._cam_kwargs["work_mode"])
        self._ctx = ob.Context()
        for spec in self._specs:
            cam = str(spec["camera"])
            self._cams[cam] = OrbbecCamera(str(spec["serial"]), camera=cam, context=self._ctx, **self._cam_kwargs)
        for cam in self._cams:
            t = threading.Thread(target=self._reader, args=(cam,), daemon=True, name=f"Orbbec-{cam}")
            t.start()
            self._threads.append(t)

    def _reader(self, cam: str) -> None:
        drv = self._cams[cam]
        while not self._stop:
            try:
                data = drv.read()
            except Exception as e:  # noqa: BLE001 - keep the others streaming
                print(f"[{self.name}] {cam} read failed: {e}")
                time.sleep(0.5)
                continue
            with self._lock:
                self._latest[cam] = data

    def step(self) -> None:
        with self._lock:
            latest = dict(self._latest)
        for cam, data in latest.items():
            if data.timestamp == self._seen.get(cam):
                continue
            self._seen[cam] = data.timestamp
            drv = self._cams[cam]
            ts = data.timestamp / 1000.0
            images = dict(data.images)
            right = images.pop("rgb_right", None)
            msg: dict = {"images": images, "timestamp": ts}
            depth = getattr(data, "depth_data", None)
            if self._stereo:
                matched = self.get_latest(f"{self._depth_topic}/{cam}") if self._subscriber is not None else None
                if matched and matched.get("depth") is not None:
                    depth = matched["depth"]
                    msg["depth_timestamp"] = matched.get("source_ts")
                now = time.time()
                if right is not None and now - self._stereo_sent.get(cam, 0.0) >= self._stereo_period:
                    self._stereo_sent[cam] = now
                    self.publish(f"stereo_{cam}", {"left": images["rgb"], "right": right, "timestamp": ts,
                                                   "calibration": drv.stereo_calibration}, ts=ts)
            if depth is not None:
                msg["depth_data"] = depth
            if drv.intrinsic_data is not None:
                msg["intrinsics"] = drv.intrinsic_data
            if drv.extrinsics is not None:
                msg["extrinsics"] = drv.extrinsics
            w = self._writers.get(cam)
            if w is not None and w.is_open:
                w.write("rgb", ts, msg)
            self.publish(cam, msg, ts=ts)

    # One writer per camera (the node's own writer is unused).
    def start_recording(self, save_dir: str) -> None:
        if self._record:
            from robots_realtime.runtime.recording import RgbdWriter

            for cam in self._cams or [str(c["camera"]) for c in self._specs]:
                w = RgbdWriter(fps=self._fps)
                w.open(save_dir, f"{self.name}_{cam}")
                self._writers[cam] = w
        self._recording = True

    def stop_recording(self) -> str:
        self._recording = False
        for w in self._writers.values():
            if w.is_open:
                w.close()
        self._writers = {}
        return ""

    def cleanup(self) -> None:
        self._stop = True
        for t in self._threads:
            t.join(timeout=1.0)
        for cam in self._cams.values():
            cam.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        keys = ("name", "cameras", "width", "height", "fps", "enable_depth", "work_mode", "rig_yaml",
                "poll_freq", "publish_freq", "record", "stereo", "stereo_freq", "depth_topic")
        return {k: params[k] for k in keys if k in params}


__all__ = ["OrbbecRigNode"]
