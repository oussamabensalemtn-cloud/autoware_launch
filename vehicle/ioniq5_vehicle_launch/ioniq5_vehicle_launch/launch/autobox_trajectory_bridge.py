#!/usr/bin/env python3
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import array
import math
import sys
import time

import numpy as np
from scipy.linalg import solveh_banded

TWO_PI = 2.0 * math.pi
MIN_SEGMENT = 1e-3  # [m] shorter steps are repeated points (e.g. after the stop point)
PENALTY_ORDER = 3   # penalize the change of curvature: straights and constant curvature stay exact
FIELDS = ("t", "x", "y", "yaw", "v", "a", "s", "kappa")


def wrap_0_2pi(angle):
    wrapped = np.mod(angle, TWO_PI)
    wrapped[wrapped >= TWO_PI] = 0.0  # np.mod(-1e-17, 2pi) rounds to exactly 2pi
    return wrapped


def yaw_from_quaternion(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def arc_length(x, y):
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def kinematic_time(s, v, v_min):
    v_mean = np.maximum(0.5 * (np.abs(v[:-1]) + np.abs(v[1:])), v_min)
    return np.concatenate(([0.0], np.cumsum(np.diff(s) / v_mean)))


def first_direction_change(v, eps=1e-3):
    """Index of the first point driving in the opposite direction (len(v) if there is none)."""
    moving = np.flatnonzero(np.abs(v) > eps)
    if moving.size == 0:
        return len(v)
    flips = moving[np.sign(v[moving]) != np.sign(v[moving[0]])]
    return int(flips[0]) if flips.size else len(v)


def whittaker_matrix(n, lam, order=PENALTY_ORDER):
    """I + lam * D^T D (D: order-th differences) in upper banded form for solveh_banded."""
    c = np.array([(-1.0) ** (order - k) * math.comb(order, k) for k in range(order + 1)])
    rows = n - order
    ab = np.zeros((order + 1, n))
    for m in range(order + 1):
        diag = np.zeros(n - m)
        for p in range(order + 1 - m):
            diag[p:p + rows] += c[p] * c[p + m]
        ab[order - m, m:] = lam * diag
    ab[order] += 1.0
    return ab


def smooth_xy(xy, h, length):
    """Whittaker smoother on a uniform grid; shapes with wavelength 'length' [m] are damped to half."""
    lam = (length / (TWO_PI * h)) ** (2 * PENALTY_ORDER)
    return solveh_banded(whittaker_matrix(len(xy), lam), xy)


def process(x, y, yaw_planner, v, a, ds, smoothing_length, max_deviation, v_min, max_length):
    """Smooth the trajectory and resample it at uniform arc length. Returns (fields, info) or None.
    max_deviation is only reported (info), the smoothing length is fixed."""
    n = first_direction_change(v)
    x, y, yaw_planner, v, a = x[:n], y[:n], yaw_planner[:n], v[:n], a[:n]

    s = arc_length(x, y)
    last = np.concatenate((np.diff(s) > MIN_SEGMENT, [True]))  # last point of each group of repeated points
    s, x, y, yaw_planner, v, a = s[last] - s[last][0], x[last], y[last], yaw_planner[last], v[last], a[last]
    if len(s) < 2:
        return None

    # planner yaw is the vehicle heading; opposite to the point order when reversing
    reverse = bool(np.mean(np.cos(yaw_planner[:-1] - np.arctan2(np.diff(y), np.diff(x)))) < 0.0)
    flip = math.pi if reverse else 0.0

    s_end = s[-1]
    n_u = int(math.ceil(s_end / ds)) + 1
    if n_u < 4 * PENALTY_ORDER:
        # too short to smooth (right before the goal): straight segments, curvature 0
        heading = np.arctan2(np.diff(y), np.diff(x))
        fields = {"t": kinematic_time(s, v, v_min), "x": x, "y": y,
                  "yaw": wrap_0_2pi(np.append(heading, heading[-1]) + flip),
                  "v": v, "a": a, "s": s, "kappa": np.zeros_like(s)}
        return fields, {"deviation": 0.0, "reverse": reverse}

    s_u = np.linspace(0.0, s_end, n_u)
    h = s_u[1]
    xy_ref = np.column_stack((np.interp(s_u, s, x), np.interp(s_u, s, y)))

    xy = smooth_xy(xy_ref, h, smoothing_length)
    deviation = float(np.max(np.hypot(xy[:, 0] - xy_ref[:, 0], xy[:, 1] - xy_ref[:, 1])))

    dx = np.gradient(xy[:, 0], h, edge_order=2)
    dy = np.gradient(xy[:, 1], h, edge_order=2)
    ddx = np.gradient(dx, h, edge_order=2)
    ddy = np.gradient(dy, h, edge_order=2)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    if reverse:
        kappa = -kappa  # steering curvature needs same sign for the same steering direction (in reverse the orders of points and with it the sign are reversed)

    s_out = arc_length(xy[:, 0], xy[:, 1])
    v_u = np.interp(s_u, s, v)
    fields = {"t": kinematic_time(s_out, v_u, v_min), "x": xy[:, 0], "y": xy[:, 1],
              "yaw": wrap_0_2pi(np.arctan2(dy, dx) + flip), "v": v_u,
              "a": np.interp(s_u, s, a), "s": s_out, "kappa": kappa}

    keep = s_out <= max_length
    keep[:2] = True
    fields = {name: val[keep] for name, val in fields.items()}
    return fields, {"deviation": deviation, "reverse": reverse}


def selftest():
    rng = np.random.default_rng(0)
    cfg = {"ds": 0.2, "smoothing_length": 8.0, "max_deviation": 0.10, "v_min": 0.1, "max_length": 1e9}
    results = []

    def check(name, passed, detail):
        results.append(bool(passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {name:32s} {detail}")

    def run(x, y, v=5.0, yaw=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        v = np.broadcast_to(np.asarray(v, float), (n,)).copy()
        if yaw is None:
            yaw = np.arctan2(np.gradient(y), np.gradient(x))
        return process(x, y, np.asarray(yaw, float), v, np.zeros(n), **cfg)

    def invariants(f):
        return bool(np.all(np.diff(f["s"]) > 0) and np.all(np.diff(f["t"]) > 0)
                    and np.all((f["yaw"] >= 0) & (f["yaw"] < TWO_PI))
                    and all(np.all(np.isfinite(val)) for val in f.values()))

    th = np.arange(0.0, math.pi / 2, 0.5 / 20.0)
    noise = rng.normal(0.0, 0.01, (2, th.size))
    f, info = run(20 * np.cos(th) + noise[0], 20 * np.sin(th) + noise[1])
    inner = (f["s"] > 5.0) & (f["s"] < f["s"][-1] - 5.0)
    e_in, e_all = np.abs(f["kappa"][inner] - 0.05).max(), np.abs(f["kappa"] - 0.05).max()
    check("circle R=20, 0.5 m, 1 cm noise", e_in < 0.01 and e_all < 0.03 and invariants(f),
          f"kappa error interior {e_in:.4f}, incl. ends {e_all:.4f}, deviation {info['deviation']:.3f} m")

    f, _ = run(20 * np.cos(-th), 20 * np.sin(-th))
    e_in = np.abs(f["kappa"][(f["s"] > 5.0) & (f["s"] < f["s"][-1] - 5.0)] + 0.05).max()
    e_all = np.abs(f["kappa"] + 0.05).max()
    check("clockwise circle R=20", e_in < 0.002 and e_all < 0.02 and invariants(f),
          f"kappa error interior {e_in:.4f}, incl. ends {e_all:.4f}")

    xs = np.concatenate((np.arange(0.0, 30.0 + 1e-9, 0.5), np.full(5, 30.0)))
    vs = np.concatenate((np.linspace(5.0, 0.0, 61), np.zeros(5)))
    f, _ = run(xs, np.zeros_like(xs), v=vs)
    check("straight + repeated stop point",
          np.abs(f["kappa"]).max() < 1e-9 and abs(f["s"][-1] - 30.0) < 1e-6 and f["v"][-1] == 0.0
          and invariants(f), f"length {f['s'][-1]:.3f} m, v_end {f['v'][-1]}, t_end {f['t'][-1]:.1f} s")

    phi = np.arange(0.0, 2.5, 0.05)
    s1 = np.arange(0.0, 20.0, 0.5)
    f, info = run(np.concatenate((s1, 20.0 + 10.0 * np.sin(phi))),
                  np.concatenate((np.zeros_like(s1), 10.0 - 10.0 * np.cos(phi))))
    k_straight = np.abs(f["kappa"][f["s"] < 10.0]).max()
    k_arc = np.abs(f["kappa"][(f["s"] > 28.0) & (f["s"] < f["s"][-1] - 5.0)] - 0.1).max()
    check("straight -> R=10 arc", k_straight < 0.005 and k_arc < 0.01 and invariants(f),
          f"deviation {info['deviation']:.3f} m (info only), "
          f"kappa error straight {k_straight:.4f}, arc {k_arc:.4f}")

    xr = np.arange(0.0, 10.0 + 1e-9, 0.5)
    f, info = run(xr, np.zeros_like(xr), v=-1.0, yaw=np.full(xr.size, math.pi))
    check("reverse: yaw = vehicle heading", info["reverse"] and np.abs(f["yaw"] - math.pi).max() < 1e-9
          and invariants(f), f"yaw {f['yaw'][0]:.4f} rad")


    th = np.arange(0.0, math.pi / 2, 0.5 / 20.0)
    xr, yr = 20 * np.cos(-th), 20 * np.sin(-th)          # driven clockwise (backwards, steering left)
    heading = np.arctan2(np.gradient(yr), np.gradient(xr)) + math.pi
    f, info = run(xr, yr, v=-1.0, yaw=heading)
    inner = (f["s"] > 5.0) & (f["s"] < f["s"][-1] - 5.0)
    dyaw_ds = np.gradient(np.unwrap(f["yaw"]), f["s"])
    e_k = np.abs(f["kappa"][inner] - 0.05).max()
    e_rate = np.abs(dyaw_ds[inner] - np.sign(f["v"][inner]) * f["kappa"][inner]).max()
    check("reverse, steering left: kappa > 0", info["reverse"] and e_k < 0.002 and e_rate < 0.002
          and np.all(f["v"] < 0) and invariants(f),
          f"kappa error {e_k:.4f}, yaw-rate consistency {e_rate:.4f}")
    

    xd = np.concatenate((np.arange(0.0, 10.0 + 1e-9, 0.5), np.arange(9.5, -1e-9, -0.5)))
    vd = np.concatenate((np.full(21, 1.0), np.full(20, -1.0)))
    f, _ = run(xd, np.zeros_like(xd), v=vd, yaw=np.zeros(xd.size))
    check("cut at direction change", abs(f["s"][-1] - 10.0) < 1e-6 and invariants(f),
          f"length {f['s'][-1]:.3f} m (expected 10)")

    sw = np.arange(0.0, 20.0 + 1e-9, 0.5)
    f, _ = run(sw * math.cos(-0.01), sw * math.sin(-0.01))
    check("yaw just below 2pi", np.abs(f["yaw"] - (TWO_PI - 0.01)).max() < 1e-6 and invariants(f),
          f"yaw {f['yaw'][0]:.4f} rad")

    th = np.arange(0.0, 6.0, 0.2 / 50.0)
    t0 = time.perf_counter()
    run(50 * np.cos(th), 50 * np.sin(th), v=10.0)
    print(f"[INFO] {th.size} points processed in {1e3 * (time.perf_counter() - t0):.1f} ms")

    print(f"{sum(results)}/{len(results)} checks passed")
    return all(results)


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from autoware_planning_msgs.msg import Trajectory
    from ioniq5_vehicle_launch.msg import ControllerTrajectory

    class TrajectoryBridge(Node):
        def __init__(self):
            super().__init__("autobox_trajectory_bridge")
            p = self.declare_parameter
            in_topic = p("input_topic", "/planning/trajectory").value
            out_topic = p("output_topic", "/HI5/autoware_controller_trajectory").value
            self.cfg = {
                "ds": float(p("resample_ds", 0.2).value),
                "smoothing_length": float(p("smoothing_length_m", 8.0).value),
                "max_deviation": float(p("max_deviation_m", 0.10).value),
                "v_min": float(p("time_min_velocity", 0.1).value),
                "max_length": float(p("max_output_length_m", 200.0).value),
            }
            if min(self.cfg.values()) <= 0.0:
                raise ValueError(f"all parameters must be > 0: {self.cfg}")

            qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
            self.pub = self.create_publisher(ControllerTrajectory, out_topic, qos)
            self.create_subscription(Trajectory, in_topic, self.on_trajectory, qos)
            self.get_logger().info(f"{in_topic} -> {out_topic}, {self.cfg}")

        def on_trajectory(self, msg):
            t0 = time.perf_counter()
            if len(msg.points) < 2:
                self.get_logger().warn("trajectory has fewer than 2 points, skipped", throttle_duration_sec=1.0)
                return
            raw = np.array(
                [(pt.pose.position.x, pt.pose.position.y, pt.pose.orientation.x, pt.pose.orientation.y,
                  pt.pose.orientation.z, pt.pose.orientation.w, pt.longitudinal_velocity_mps, pt.acceleration_mps2)
                 for pt in msg.points],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(raw)):
                self.get_logger().warn("trajectory contains NaN/inf, skipped", throttle_duration_sec=1.0)
                return

            x, y, qx, qy, qz, qw, v, a = raw.T
            result = process(x, y, yaw_from_quaternion(qx, qy, qz, qw), v, a, **self.cfg)
            if result is None:
                self.get_logger().warn("all trajectory points at one position, skipped", throttle_duration_sec=1.0)
                return
            fields, info = result

            out = ControllerTrajectory()
            out.header = msg.header
            for name in FIELDS:
                setattr(out, name, array.array("d", np.ascontiguousarray(fields[name], dtype=np.float64).tobytes()))
            self.pub.publish(out)

            if info["deviation"] > self.cfg["max_deviation"]:
                self.get_logger().warn(
                    f"smoothed path deviates {info['deviation']:.2f} m from Autoware "
                    f"(warning threshold {self.cfg['max_deviation']} m)", throttle_duration_sec=2.0)
            dt_ms = 1e3 * (time.perf_counter() - t0)
            if dt_ms > 30.0:
                self.get_logger().warn(f"{len(msg.points)} points took {dt_ms:.1f} ms", throttle_duration_sec=2.0)

    rclpy.init()
    node = TrajectoryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()