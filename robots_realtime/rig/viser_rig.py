"""RigViser — the live 3D view and the STOP button for the dionysus rig.

Everything is drawn in the rig's world frame (the calibration checkerboard):

- the table plane, the TCP workspace box and the world axes;
- both UR5e arms at their calibrated bases, live from ``actual_q``, plus a
  translucent ghost of each arm's commanded setpoint;
- the four cameras as frustums at their calibrated poses (wrist frustums follow
  the arm), each with a live thumbnail and an optional back-projected depth
  cloud — the quickest visual check that calibration and depth alignment agree;
- what GaP is about to do (planned TCP paths, green when they passed the safety
  gates, red when refused; IK / grasp targets; perceived object points) and the
  graph node running.

GUI: a red **STOP** (latch + REST program stop on both arms), **Reset latch**
(confirmation modal; motion still needs a fresh bring-up), **Approve / Reject**
for the next trajectory when GaP runs with ``GAP_UR_CONFIRM=1``, and a status
panel (latch, per-arm driving/safety, GaP status).

This class never raises into its caller: a broken view must not take the
control loop down with it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from robots_realtime.rig.dionysus import Rig, matrix_to_quat_wxyz

log = logging.getLogger(__name__)

_RZ_PI = np.diag([-1.0, -1.0, 1.0, 1.0])  # UR controller "base" -> URDF "base_link"


def _wxyz(T: np.ndarray) -> np.ndarray:
    return matrix_to_quat_wxyz(np.asarray(T)[:3, :3])


def _safe(fn):
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception:
            if not getattr(self, "_warned", False):
                log.exception("[RigViser] %s failed (further errors suppressed)", fn.__name__)
                self._warned = True
            return None

    return wrapper


class RigViser:
    """Viser server for the rig.

    Args:
        rig: The rig model (frames, workspace, cameras).
        port: Viser port (open remotely with ``ssh -L 8080:localhost:8080``).
        on_stop: Called with a reason when STOP is pressed (from a viser thread).
        on_reset: Called when the latch reset is confirmed; returns a message.
        cloud_stride: Pixel stride for depth back-projection.
    """

    def __init__(
        self,
        rig: Rig,
        port: int = 8080,
        on_stop: Callable[[str], None] | None = None,
        on_reset: Callable[[], str] | None = None,
        cloud_stride: int = 4,
        urdf: Any = None,
    ) -> None:
        import viser

        self.rig = rig
        self._on_stop = on_stop
        self._on_reset = on_reset
        self._stride = int(cloud_stride)
        self._lock = threading.Lock()
        self._ack: dict | None = None
        self._awaiting: dict | None = None
        self._gap_seq = -1
        self._last_cam_t: dict[str, float] = {}
        self._warned = False
        self.server = viser.ViserServer(host="0.0.0.0", port=port, label="dionysus rig", verbose=False)
        self._build_static()
        self._build_gui()
        self._build_arms(urdf)

    # ── construction ──────────────────────────────────────────────────────

    def _build_static(self) -> None:
        sc = self.server.scene
        rig = self.rig
        sc.add_frame("/world", axes_length=0.15, axes_radius=0.004)
        lo, hi = rig.workspace_min, rig.workspace_max
        center = (lo + hi) / 2.0
        size = hi - lo
        sc.add_box("/world/table", color=(205, 195, 180),
                   dimensions=(size[0] + 0.4, size[1] + 0.4, 0.002),
                   position=(center[0], center[1], rig.surface_z - 0.001), opacity=0.6)
        self._ws_box = sc.add_box("/world/workspace", color=(60, 140, 220), dimensions=tuple(size),
                                  position=tuple(center), wireframe=True)
        for arm in rig.ordered_arms():
            T = arm.world_T_base
            sc.add_frame(f"/world/bases/{arm.name}", axes_length=0.08, axes_radius=0.003,
                         position=T[:3, 3], wxyz=_wxyz(T))
            sc.add_label(f"/world/bases/{arm.name}/label", f"{arm.name} (arm {arm.arm_id})",
                         position=(0.0, 0.0, 0.05))
        self._frustums: dict[str, Any] = {}
        self._clouds: dict[str, Any] = {}
        for cam in rig.cameras.values():
            self._frustums[cam.name] = sc.add_camera_frustum(
                f"/world/cams/{cam.name}", fov=1.0, aspect=848 / 530, scale=0.06,
                color=(240, 140, 30) if cam.mount == "fixed" else (200, 60, 200),
            )
            if cam.mount == "fixed":
                self._set_pose(self._frustums[cam.name], cam.T)
        self._plans = sc.add_frame("/world/gap", show_axes=False)

    def _build_gui(self) -> None:
        gui = self.server.gui
        self._stop_btn = gui.add_button("STOP", color="red", icon=None,
                                        hint="Latch + stop the program on both arms")
        self._stop_btn.on_click(lambda _: self._press_stop("viser STOP button"))
        self._status = gui.add_markdown("**latch:** ?")
        with gui.add_folder("Trajectory approval"):
            self._approval_md = gui.add_markdown("_no trajectory waiting_")
            self._approve_btn = gui.add_button("Approve", color="green", disabled=True)
            self._reject_btn = gui.add_button("Reject", color="orange", disabled=True)
            self._approve_btn.on_click(lambda _: self._answer(True))
            self._reject_btn.on_click(lambda _: self._answer(False))
        with gui.add_folder("View"):
            self._show_clouds = gui.add_checkbox("Depth clouds", True)
            self._show_ghost = gui.add_checkbox("Commanded ghost", True)
            self._cloud_hz = gui.add_slider("Camera Hz", 0.5, 10.0, 0.5, 2.0)
        with gui.add_folder("Latch", expand_by_default=False):
            reset = gui.add_button("Reset latch…", color="gray")
            reset.on_click(lambda ev: self._confirm_reset(ev.client))

    def _build_arms(self, urdf: Any) -> None:
        self._arms: dict[str, Any] = {}
        self._ghosts: dict[str, Any] = {}
        self._tcp_frames: dict[str, Any] = {}
        sc = self.server.scene
        try:
            from viser.extras import ViserUrdf

            if urdf is None:
                from robot_descriptions.loaders.yourdfpy import load_robot_description

                urdf = load_robot_description("ur5e_description")
        except Exception as e:
            log.warning("[RigViser] no UR5e URDF (%s); drawing arms as polylines", e)
            urdf = None
        for arm in self.rig.ordered_arms():
            base_link = arm.world_T_base @ _RZ_PI
            for kind, store, color in (("live", self._arms, None), ("ghost", self._ghosts, (0.2, 0.8, 1.0, 0.35))):
                root = f"/world/arms/{arm.name}_{kind}"
                sc.add_frame(root, show_axes=False, position=base_link[:3, 3], wxyz=_wxyz(base_link))
                if urdf is not None:
                    store[arm.name] = ViserUrdf(self.server, urdf, root_node_name=root,
                                                mesh_color_override=color)
                else:
                    store[arm.name] = None
            self._tcp_frames[arm.name] = sc.add_frame(f"/world/tcp/{arm.name}", axes_length=0.05,
                                                      axes_radius=0.002)

    # ── GUI callbacks ─────────────────────────────────────────────────────

    def _press_stop(self, reason: str) -> None:
        log.error("[RigViser] STOP pressed")
        if self._on_stop is not None:
            self._on_stop(reason)

    def _answer(self, approved: bool) -> None:
        with self._lock:
            if self._awaiting is None:
                return
            self._ack = {"id": int(self._awaiting["id"]), "approved": bool(approved)}
        self._approve_btn.disabled = True
        self._reject_btn.disabled = True
        self._approval_md.content = f"_{'approved' if approved else 'rejected'} #{self._ack['id']}_"

    def _confirm_reset(self, client) -> None:
        if client is None:
            return
        with client.gui.add_modal("Reset the stop latch?") as modal:
            client.gui.add_markdown(
                "Clears the latch only if both arms are safety **NORMAL** with their program **STOPPED**. "
                "Motion still needs a fresh bring-up (restart the session)."
            )
            yes = client.gui.add_button("Reset", color="red")
            no = client.gui.add_button("Cancel")

            @yes.on_click
            def _(_):
                msg = self._on_reset() if self._on_reset is not None else "no reset handler"
                self._status.content = f"**reset:** {msg}"
                modal.close()

            @no.on_click
            def _(_):
                modal.close()

    def take_ack(self) -> dict | None:
        """The operator's answer to the pending approval (consumed once)."""
        with self._lock:
            ack, self._ack = self._ack, None
        return ack

    # ── updates ───────────────────────────────────────────────────────────

    @staticmethod
    def _set_pose(handle, T: np.ndarray) -> None:
        handle.position = np.asarray(T)[:3, 3]
        handle.wxyz = _wxyz(T)

    @_safe
    def update_arms(self, q_by_arm: dict[str, np.ndarray], setpoint_by_arm: dict[str, np.ndarray]) -> None:
        for name, q in q_by_arm.items():
            arm = self.rig.arms.get(name)
            if arm is None or q is None:
                continue
            q6 = np.asarray(q, dtype=np.float64)[:6]
            if self._arms.get(name) is not None:
                self._arms[name].update_cfg(q6)
            else:  # no URDF: the arm as a polyline through its DH frames
                pts = arm.link_points_world(q6).astype(np.float32)
                self.server.scene.add_line_segments(
                    f"/world/arms/{name}_poly", points=np.stack([pts[:-1], pts[1:]], axis=1),
                    colors=(90, 90, 90), thickness=0.03)
            self._set_pose(self._tcp_frames[name], arm.tcp_world(q6))
            ghost = self._ghosts.get(name)
            sp = setpoint_by_arm.get(name)
            if ghost is not None:
                ghost.show_visual = bool(self._show_ghost.value) and sp is not None
                if sp is not None:
                    ghost.update_cfg(np.asarray(sp, dtype=np.float64)[:6])
        for cam in self.rig.cameras.values():
            if cam.mount != "fixed" and cam.mount in q_by_arm and q_by_arm[cam.mount] is not None:
                T = cam.world_pose(self.rig.arms, q_by_arm)
                if T is not None:
                    self._set_pose(self._frustums[cam.name], T)

    def camera_due(self, name: str) -> bool:
        now = time.time()
        if now - self._last_cam_t.get(name, 0.0) < 1.0 / max(float(self._cloud_hz.value), 0.1):
            return False
        self._last_cam_t[name] = now
        return True

    @_safe
    def update_camera(self, name: str, rgb: np.ndarray | None, depth: np.ndarray | None,
                      K: np.ndarray | None, world_T_cam: np.ndarray | None) -> None:
        fr = self._frustums.get(name)
        if fr is None or rgb is None:
            return
        h, w = rgb.shape[:2]
        if K is not None:
            fr.fov = float(2.0 * np.arctan2(h / 2.0, float(K[1, 1])))
            fr.aspect = float(w) / float(h)
        fr.image = rgb[::4, ::4]
        if world_T_cam is not None:
            self._set_pose(fr, world_T_cam)
        if depth is None or K is None or world_T_cam is None or not self._show_clouds.value:
            if name in self._clouds:
                self._clouds[name].visible = False
            return
        s = self._stride
        d = np.asarray(depth, dtype=np.float32)[::s, ::s]
        vv, uu = np.mgrid[0:h:s, 0:w:s]
        m = d > 0
        z = d[m]
        x = (uu[m] - K[0, 2]) * z / K[0, 0]
        y = (vv[m] - K[1, 2]) * z / K[1, 1]
        pts = np.stack([x, y, z], axis=1) @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]
        cols = rgb[::s, ::s][m]
        if name in self._clouds:
            self._clouds[name].points = pts.astype(np.float32)
            self._clouds[name].colors = cols
            self._clouds[name].visible = True
        else:
            self._clouds[name] = self.server.scene.add_point_cloud(
                f"/world/clouds/{name}", points=pts.astype(np.float32), colors=cols, point_size=0.003,
            )

    @_safe
    def update_status(self, latch: dict | None, arms: dict[str, dict], gap_status: str = "") -> None:
        lines = [f"**latch:** {'🛑 SET — ' + str(latch.get('reason')) if latch else '✅ clear'}"]
        for name, a in arms.items():
            lines.append(
                f"- **{name}**: {'DRIVING' if a.get('driving') else 'read-only'}"
                f" · safety {a.get('safety_mode', '?')} · runtime {a.get('runtime_state', '?')}"
            )
        if gap_status:
            lines.append(f"\n**GaP:** {gap_status}")
        self._status.content = "\n".join(lines)

    @_safe
    def update_from_gap(self, viz: dict | None) -> None:
        """Draw GaP's plans/targets/points; ``viz["seq"]`` bumps when they change."""
        if not viz:
            return
        seq = int(viz.get("seq", -1))
        awaiting = viz.get("awaiting_approval")
        with self._lock:
            new_wait = awaiting is not None and (self._awaiting is None or awaiting.get("id") != self._awaiting.get("id"))
            self._awaiting = awaiting
        if new_wait:
            self._approval_md.content = f"**#{awaiting['id']}** {awaiting.get('summary', '')}"
            self._approve_btn.disabled = False
            self._reject_btn.disabled = False
        elif awaiting is None:
            self._approve_btn.disabled = True
            self._reject_btn.disabled = True
        if seq == self._gap_seq:
            return
        self._gap_seq = seq
        sc = self.server.scene
        self._plans.remove()
        self._plans = sc.add_frame("/world/gap", show_axes=False)
        for i, plan in enumerate(viz.get("plans") or []):
            path = np.asarray(plan.get("tcp_path") or [], dtype=np.float32).reshape(-1, 3)
            if len(path) < 2:
                continue
            ok = bool(plan.get("ok", True))
            segs = np.stack([path[:-1], path[1:]], axis=1)
            sc.add_line_segments(f"/world/gap/plan_{i}", points=segs,
                                 colors=(40, 200, 80) if ok else (230, 40, 40), thickness=0.004)
            sc.add_label(f"/world/gap/plan_{i}/label", str(plan.get("label", "")), position=tuple(path[-1]))
        for i, tgt in enumerate(viz.get("targets") or []):
            p = np.asarray(tgt.get("pose"), dtype=np.float64).reshape(-1)
            if p.size < 7:
                continue
            sc.add_frame(f"/world/gap/target_{i}", axes_length=0.06, axes_radius=0.003,
                         position=p[:3], wxyz=p[3:7])
            if tgt.get("name"):
                sc.add_label(f"/world/gap/target_{i}/label", str(tgt["name"]))
        for i, pc in enumerate(viz.get("points") or []):
            xyz = np.asarray(pc.get("xyz") or [], dtype=np.float32).reshape(-1, 3)
            if len(xyz) == 0:
                continue
            col = tuple(int(c) for c in (pc.get("color") or (255, 220, 0)))
            sc.add_point_cloud(f"/world/gap/points_{i}", points=xyz, colors=col, point_size=0.006)
        violated = bool(viz.get("workspace_violation"))
        self._ws_box.color = (230, 40, 40) if violated else (60, 140, 220)

    def close(self) -> None:
        try:
            self.server.stop()
        except Exception:
            pass


__all__ = ["RigViser"]
