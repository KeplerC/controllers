"""URBimanualClientAgent — the dionysus rig's side of the GaP msgpack bridge.

Runs as an ``AgentNode`` (fixed rate, ~100 Hz) inside ``rr-session``. Each tick:

1. reads both arms' joint state and the latest frame of each camera off the bus;
2. sends GaP (the TCP server ``gap.envs.ur5e_bimanual_real_env`` binds) one
   observation and receives the latest action — the same lockstep protocol the
   Franka client speaks (4-byte length + msgpack-numpy), extended for two arms;
3. returns per-arm 7-vectors (6 joints + gripper open fraction) that AgentNode
   publishes as ``{name}/{arm}_pos`` for the RobotNodes.

It also hosts the rig's viser view (:class:`robots_realtime.rig.viser_rig.RigViser`)
with the STOP button, so the view and the stop work in capture-only sessions
(``gap_port: null``) as well as GaP rollouts.

Observation sent to GaP (string keys)::

    {"timestamp", "seq",
     "right"/"left": {"joint_pos": f32[7], "tcp_world": f32[7] (xyz + wxyz),
                      "setpoint": f32[6], "driving": int, "latched": int,
                      "safety_mode": int, "runtime_state": int},
     "<camera>": {"images": {"rgb"}, "depth_data": f32[H,W] (m), "timestamp",
                  "intrinsics": {"left": {"intrinsics_matrix"}}, "pose": f32[7],
                  "pose_mat": f32[4,4]},          # camera->world, OpenCV optical
     "rig": {...}  (static rig description, ~1 Hz),
     "stop": {"reason"} | absent, "viz_ack": {"id","approved"} | absent}

Action expected back::

    {"timestamp", "right"/"left": {"joint_pos": [6], "gripper": open fraction},
     "viz": {...} | absent}

Safety: on no reply, a stale reply (older than ``stale_after_s``), STOP, the
latch, or a released deadman, every arm is commanded to *hold where it is*.
There is no fallback target of any kind.
"""

from __future__ import annotations

import glob
import logging
import os
import socket
import struct
import threading
import time
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np

from robots_realtime.agents.agent import Agent
from robots_realtime.rig import stop as rigstop
from robots_realtime.rig.dionysus import load_rig, pose7_wxyz

msgpack_numpy.patch()
log = logging.getLogger(__name__)


def _str_keys(obj: Any) -> Any:
    """bytes -> str for dict keys (decoded with raw=True for msgpack-numpy)."""
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _str_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_str_keys(v) for v in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode()
        except UnicodeDecodeError:
            return obj
    return obj


class _BridgeClient:
    """Blocking framed msgpack client with timeouts and reconnect."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.5, retry_s: float = 1.0) -> None:
        self.host, self.port = host, int(port)
        self.timeout_s, self.retry_s = float(timeout_s), float(retry_s)
        self._sock: socket.socket | None = None
        self._next_try = 0.0

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _connect(self) -> bool:
        if time.time() < self._next_try:
            return False
        self._next_try = time.time() + self.retry_s
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            s.settimeout(self.timeout_s)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            log.info("[ur_client] connected to GaP %s:%d", self.host, self.port)
            return True
        except OSError:
            return False

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("server closed")
            buf.extend(chunk)
        return bytes(buf)

    def request(self, obs: dict) -> dict | None:
        if self._sock is None and not self._connect():
            return None
        try:
            payload = msgpack.packb(obs, use_bin_type=True)
            self._sock.sendall(struct.pack("!I", len(payload)) + payload)  # type: ignore[union-attr]
            (n,) = struct.unpack("!I", self._recv_exact(4))
            return _str_keys(msgpack.unpackb(self._recv_exact(n), raw=True))
        except (OSError, ConnectionError, ValueError) as e:
            log.warning("[ur_client] bridge error (%s); holding and reconnecting", e)
            self.close()
            return None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None


class _Deadman:
    """PCsensor foot pedal as a deadman: True while any key is held."""

    def __init__(self, pattern: str = "/dev/input/by-id/*FootSwitch*event-kbd") -> None:
        self.pressed = False
        self.ok = False
        paths = sorted(glob.glob(pattern))
        if not paths:
            log.error("[deadman] no device matches %s — deadman reads as RELEASED", pattern)
            return
        try:
            self._fd = os.open(paths[0], os.O_RDONLY)
        except OSError as e:
            log.error("[deadman] cannot open %s (%s) — deadman reads as RELEASED", paths[0], e)
            return
        self.ok = True
        self._down: set[int] = set()
        threading.Thread(target=self._loop, daemon=True, name="Deadman").start()

    def _loop(self) -> None:
        fmt = "llHHi"
        size = struct.calcsize(fmt)
        while True:
            try:
                data = os.read(self._fd, size)
            except OSError:
                self.ok = False
                self.pressed = False
                return
            if len(data) < size:
                continue
            _, _, etype, code, value = struct.unpack(fmt, data)
            if etype == 1:  # EV_KEY
                if value:
                    self._down.add(code)
                else:
                    self._down.discard(code)
                self.pressed = bool(self._down)


class URBimanualClientAgent(Agent):
    """GaP bridge client + viser view + stop for the bimanual UR5e rig.

    Args:
        rig_yaml: Rig model path (default ``configs/ur5e_bimanual/rig.yaml``).
        gap_host / gap_port: GaP bridge server; ``gap_port: null`` = standalone
            (viewer + stop only, arms held).
        arms: Arm names in the observation (default: every arm in the rig).
        cameras: Camera names to forward (default: every camera in the rig).
        camera_period_s: Min period between sends of one camera to GaP.
        stale_after_s: A GaP action older than this is not executed (hold).
        viser_port: RigViser port; ``null`` disables the view.
        deadman: ``"pedal"`` to require the foot pedal held for motion.
    """

    def __init__(
        self,
        rig_yaml: str | None = None,
        gap_host: str = "127.0.0.1",
        gap_port: int | None = 9000,
        arms: list[str] | None = None,
        cameras: list[str] | None = None,
        camera_period_s: float = 0.2,
        stale_after_s: float = 0.25,
        viser_port: int | None = 8080,
        deadman: str | None = None,
        bridge_timeout_s: float = 0.5,
    ) -> None:
        self.rig = load_rig(rig_yaml)
        self.arm_names = list(arms or [a.name for a in self.rig.ordered_arms()])
        self.camera_names = list(cameras or list(self.rig.cameras))
        self.camera_period_s = float(camera_period_s)
        self.stale_after_s = float(stale_after_s)
        self._bridge = _BridgeClient(gap_host, gap_port, timeout_s=bridge_timeout_s) if gap_port else None
        self._seq = 0
        self._last_rig_t = 0.0
        self._last_cam_sent: dict[str, float] = {}
        self._last_cam_ts: dict[str, float] = {}
        self._last_status_t = 0.0
        self._lag_max: dict[str, float] = {}  # worst arm-state age (s) since the last lag report
        self._lag_report_t = 0.0
        self._stop_reason: str | None = None
        self._pending_ack: dict | None = None
        self._studio_seen: dict[str, Any] = {}
        """The last ``t`` taken from each ``studio/*`` message, so one message acts once."""
        self._gap_status = "standalone (no GaP)" if self._bridge is None else "waiting for GaP"
        self._deadman = _Deadman() if deadman == "pedal" else None
        self._deadman_was_pressed = False
        self._ips = {a.name: a.ip for a in self.rig.arms.values()}
        self.viser = None
        if viser_port:
            from robots_realtime.rig.viser_rig import RigViser

            self.viser = RigViser(self.rig, port=int(viser_port), on_stop=self.request_stop,
                                  on_reset=self.reset_latch)
            log.info("[ur_client] viser on http://localhost:%d", int(viser_port))

    # ── stop / reset ──────────────────────────────────────────────────────

    def request_stop(self, reason: str) -> None:
        """Layer 1 (latch -> drivers HOLD next tick) + Layer 2 (REST program stop)."""
        self._stop_reason = reason
        rigstop.latch_set(reason, "URBimanualClientAgent")
        threading.Thread(target=rigstop.emergency_stop, args=(self._ips, reason, "URBimanualClientAgent"),
                         daemon=True).start()

    def reset_latch(self) -> str:
        bad = []
        for name, ip in self._ips.items():
            st = rigstop.arm_status(ip)
            if st.get("safety_mode") != "NORMAL" or st.get("program_state") != "STOPPED":
                bad.append(f"{name}: program={st.get('program_state')} safety={st.get('safety_mode')}")
        if bad:
            return "refused — " + "; ".join(bad)
        rigstop.latch_clear()
        self._stop_reason = None
        return "latch cleared; restart the session to drive again"

    # ── helpers ───────────────────────────────────────────────────────────

    def _note_lag(self, now: float, states: dict) -> None:
        """Track how old each arm state is when it reaches this loop; report every 5 s."""
        for n, s in states.items():
            if isinstance(s, dict) and s.get("state_t") is not None:
                lag = now - float(np.asarray(s["state_t"]).reshape(-1)[0])
                self._lag_max[n] = max(self._lag_max.get(n, 0.0), lag)
        if now - self._lag_report_t >= 5.0 and self._lag_max:
            worst = max(self._lag_max.values())
            (log.warning if worst > 0.1 else log.info)(
                "[ur_client] arm state age (max over %.0fs): %s", now - self._lag_report_t if self._lag_report_t else 5.0,
                ", ".join(f"{n} {v * 1e3:.0f} ms" for n, v in sorted(self._lag_max.items())))
            self._lag_max, self._lag_report_t = {}, now

    @staticmethod
    def _joint_pos(state: dict | None) -> np.ndarray | None:
        if not isinstance(state, dict) or state.get("joint_pos") is None:
            return None
        return np.asarray(state["joint_pos"], dtype=np.float64).reshape(-1)

    def _arm_obs(self, name: str, state: dict) -> dict:
        jp = self._joint_pos(state)
        arm = self.rig.arms[name]

        def _i(key: str) -> int:
            v = state.get(key)
            return int(np.asarray(v).reshape(-1)[0]) if v is not None else -1

        return {
            "joint_pos": jp.astype(np.float32),
            "tcp_world": pose7_wxyz(arm.tcp_world(jp[:6])).astype(np.float32),
            "setpoint": np.asarray(state.get("setpoint", jp[:6]), dtype=np.float32)[:6],
            "driving": _i("driving"),
            "latched": _i("latched"),
            "safety_mode": _i("safety_mode"),
            "runtime_state": _i("runtime_state"),
            # the wrist's force/torque at the TCP, controller base frame (N, N m); zeros from an older driver
            "tcp_force": np.asarray(state.get("tcp_force", np.zeros(6)), dtype=np.float32).reshape(-1)[:6],
        }

    def _camera_obs(self, name: str, msg: dict, q_by_arm: dict[str, np.ndarray]) -> tuple[dict | None, Any]:
        cam = self.rig.cameras[name]
        T = cam.world_pose(self.rig.arms, q_by_arm)
        images = msg.get("images") or {}
        rgb = images.get("rgb")
        if rgb is None:
            rgb = images.get("left_rgb")
        intr = (msg.get("intrinsics") or {}).get("left") or {}
        K = intr.get("intrinsics_matrix")
        out = {
            "images": {"rgb": rgb},
            "timestamp": float(msg.get("timestamp", 0.0)),
            "intrinsics": {"left": {"intrinsics_matrix": np.asarray(K, dtype=np.float32)}} if K is not None else {},
        }
        if msg.get("depth_data") is not None:
            out["depth_data"] = np.asarray(msg["depth_data"], dtype=np.float32)
        if T is not None:
            out["pose_mat"] = T.astype(np.float32)
            out["pose"] = pose7_wxyz(T).astype(np.float32)
        return out, (rgb, out.get("depth_data"), None if K is None else np.asarray(K, dtype=np.float64), T)

    # ── Agent protocol ────────────────────────────────────────────────────

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        states = {n: obs.get(n) for n in self.arm_names}
        q_by_arm = {n: (self._joint_pos(s)[:6] if self._joint_pos(s) is not None else None) for n, s in states.items()}
        self._note_lag(now, states)
        latch = rigstop.latch_active()
        if latch and self._stop_reason is None:
            self._stop_reason = str(latch.get("reason"))

        # The hub, over the bus (``extra_topics`` in the node config): its STOP acts at once, its
        # answer to GaP's pending motion is forwarded as ``viz_ack`` exactly as the viser buttons' is.
        studio_stop = obs.get("studio_stop")
        if isinstance(studio_stop, dict) and studio_stop.get("t") != self._studio_seen.get("stop"):
            self._studio_seen["stop"] = studio_stop.get("t")
            self.request_stop(f"hub: {studio_stop.get('reason', 'STOP')}")
            latch = rigstop.latch_active()
        studio_ack = obs.get("studio_ack")
        if isinstance(studio_ack, dict) and studio_ack.get("t") != self._studio_seen.get("ack"):
            self._studio_seen["ack"] = studio_ack.get("t")
            self._pending_ack = {"id": studio_ack.get("id"), "approved": bool(studio_ack.get("approved"))}
            self._pending_ack_until = now + 1.0

        # Deadman: releasing it while armed is a STOP.
        motion_allowed = latch is None and self._stop_reason is None
        if self._deadman is not None:
            pressed = self._deadman.ok and self._deadman.pressed
            if self._deadman_was_pressed and not pressed:
                self.request_stop("deadman pedal released")
            self._deadman_was_pressed = pressed
            motion_allowed = motion_allowed and pressed

        # Build + send the GaP observation.
        reply = None
        if self._bridge is not None:
            gap_obs: dict[str, Any] = {"timestamp": now, "seq": self._seq}
            self._seq += 1
            for n, s in states.items():
                if self._joint_pos(s) is not None:
                    gap_obs[n] = self._arm_obs(n, s)
            for c in self.camera_names:
                msg = obs.get(c)
                if not isinstance(msg, dict):
                    continue
                ts = float(msg.get("timestamp", 0.0))
                fresh = ts != self._last_cam_ts.get(c)
                if fresh and now - self._last_cam_sent.get(c, 0.0) >= self.camera_period_s:
                    cam_obs, _ = self._camera_obs(c, msg, q_by_arm)
                    gap_obs[c] = cam_obs
                    self._last_cam_sent[c] = now
                    self._last_cam_ts[c] = ts
            if now - self._last_rig_t > 1.0:
                gap_obs["rig"] = self.rig.to_wire()
                self._last_rig_t = now
            if self._stop_reason is not None:
                gap_obs["stop"] = {"reason": self._stop_reason}
            if self._deadman is not None:
                gap_obs["deadman"] = int(bool(self._deadman.ok and self._deadman.pressed))
            ack = self.viser.take_ack() if self.viser is not None else None
            if ack is not None:
                self._pending_ack = ack
                self._pending_ack_until = now + 1.0  # repeat for a second; GaP dedups by id
            if self._pending_ack is not None:
                gap_obs["viz_ack"] = self._pending_ack
                if now > getattr(self, "_pending_ack_until", 0.0):
                    self._pending_ack = None
            reply = self._bridge.request(gap_obs)
            if reply is not None:
                self._gap_status = str((reply.get("viz") or {}).get("status") or "connected")
            else:
                self._gap_status = "GaP not connected — holding"

        # Decide each arm's command: GaP's target if fresh and allowed, else hold here.
        action: dict[str, Any] = {}
        fresh_reply = (
            reply is not None
            and isinstance(reply.get("timestamp"), (int, float))
            and now - float(reply["timestamp"]) <= self.stale_after_s
        )
        for n, s in states.items():
            jp = self._joint_pos(s)
            if jp is None:
                continue
            cmd = jp.copy()  # hold: 6 joints + current gripper
            arm_cmd = reply.get(n) if (fresh_reply and motion_allowed) else None
            if isinstance(arm_cmd, dict) and arm_cmd.get("joint_pos") is not None:
                tgt = np.asarray(arm_cmd["joint_pos"], dtype=np.float64).reshape(-1)[:6]
                grip = arm_cmd.get("gripper")
                if tgt.shape[0] == 6 and np.all(np.isfinite(tgt)):
                    cmd[:6] = tgt
                    if grip is not None and np.isfinite(float(grip)):
                        cmd[6] = float(np.clip(float(grip), 0.0, 1.0))
            action[n] = {"pos": cmd}
        studio_record = obs.get("studio_record")
        if isinstance(studio_record, dict) and studio_record.get("t") != self._studio_seen.get("record"):
            self._studio_seen["record"] = studio_record.get("t")
            action["_record"] = bool(studio_record.get("record"))
        if reply is not None and reply.get("viz") is not None:
            action["_viz"] = reply["viz"]  # the node publishes it as ur_client/viz for the hub

        # View.
        if self.viser is not None:
            setpoints = {n: (np.asarray(s.get("setpoint"))[:6] if isinstance(s, dict) and s.get("setpoint") is not None
                             else None) for n, s in states.items()}
            self.viser.update_arms({n: q for n, q in q_by_arm.items() if q is not None}, setpoints)
            for c in self.camera_names:
                msg = obs.get(c)
                if isinstance(msg, dict) and self.viser.camera_due(c):
                    _, (rgb, depth, K, T) = self._camera_obs(c, msg, q_by_arm)
                    self.viser.update_camera(c, rgb, depth, K, T)
            if reply is not None and reply.get("viz") is not None:
                self.viser.update_from_gap(reply["viz"])
            if now - self._last_status_t > 0.5:
                self._last_status_t = now
                arms_status = {}
                for n, s in states.items():
                    if isinstance(s, dict):
                        arms_status[n] = {
                            "driving": int(np.asarray(s.get("driving", [0])).reshape(-1)[0]),
                            "safety_mode": int(np.asarray(s.get("safety_mode", [-1])).reshape(-1)[0]),
                            "runtime_state": int(np.asarray(s.get("runtime_state", [-1])).reshape(-1)[0]),
                        }
                self.viser.update_status(latch, arms_status, self._gap_status)
        return action

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
        if self.viser is not None:
            self.viser.close()


__all__ = ["URBimanualClientAgent"]
