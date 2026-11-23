#!/usr/bin/env python3
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import math
import socket
import struct
import sys

import numpy as np

MAGIC = 0x4157544A
VERSION = 3
FIELDS = ("t", "x", "y", "yaw", "v", "a", "s", "kappa")
HEADER_FMT = "IHHHHIiI"  # magic, version, n_max, n_valid, spacing_mm, seq, stamp_sec, stamp_nanosec
TWO_PI = 2.0 * math.pi


def wrap_0_2pi(angle):
    wrapped = np.mod(angle, TWO_PI)
    wrapped[wrapped >= TWO_PI] = 0.0
    return wrapped


def packet_size(n_max):
    return struct.calcsize("<" + HEADER_FMT) + len(FIELDS) * 4 * n_max


def project_ego(f, ex, ey, eyaw, max_heading_diff=math.radians(60.0)):
    """Arc length of the car's projection onto the trajectory and its distance to it."""
    x, y, s = f["x"], f["y"], f["s"]
    sx, sy = np.diff(x), np.diff(y)
    u = np.clip(((ex - x[:-1]) * sx + (ey - y[:-1]) * sy) / np.maximum(sx * sx + sy * sy, 1e-12), 0.0, 1.0)
    d2 = (x[:-1] + u * sx - ex) ** 2 + (y[:-1] + u * sy - ey) ** 2
    aligned = np.cos(f["yaw"][:-1] - eyaw) > math.cos(max_heading_diff)
    if np.any(aligned):
        d2 = np.where(aligned, d2, np.inf)
    i = int(np.argmin(d2))
    return float(s[i] + u[i] * (s[i + 1] - s[i])), math.sqrt(float(d2[i]))


def choose_spacing(f, s_ego, n_intervals, min_spacing, max_spacing, horizon):
    """Smallest spacing >= min_spacing whose points ahead reach 'horizon' seconds of the planned motion."""
    t_ego = float(np.interp(s_ego, f["s"], f["t"]))
    s_horizon = float(np.interp(t_ego + horizon, f["t"], f["s"]))
    spacing = min(max((s_horizon - s_ego) / n_intervals, min_spacing), max_spacing)
    return round(spacing, 3)  # whole mm, matches spacing_mm in the header


def interp_fields(f, targets, v_min):
    """All fields at arc lengths 'targets'; straight extrapolation before the trajectory start."""
    s = f["s"]
    w = {name: np.interp(targets, s, f[name]) for name in ("t", "x", "y", "v", "a", "kappa")}
    w["yaw"] = wrap_0_2pi(np.interp(targets, s, np.unwrap(f["yaw"])))

    before = targets < s[0]
    if np.any(before):
        ds = targets[before] - s[0]  # negative
        dx, dy = f["x"][1] - f["x"][0], f["y"][1] - f["y"][0]
        seg = max(math.hypot(dx, dy), 1e-9)
        w["x"][before] = f["x"][0] + ds * dx / seg
        w["y"][before] = f["y"][0] + ds * dy / seg
        w["t"][before] = f["t"][0] + ds / max(abs(f["v"][0]), v_min)
    return w


def sample_window(f, s_ego, spacing, n_max, n_behind, behind_spacing, v_min):
    """n_behind points behind the car, the car itself (index n_behind), then points every 'spacing' m.
    s and t are 0 at the first point."""
    s_end = f["s"][-1]
    n_ahead = n_max - n_behind  # includes the car's point
    behind = s_ego - behind_spacing * np.arange(n_behind, 0, -1)
    n_steps = int(math.floor((s_end - s_ego) / spacing + 1e-9))
    ahead = s_ego + spacing * np.arange(min(n_steps + 1, n_ahead))
    if len(ahead) < n_ahead and s_end - ahead[-1] > 1e-3:
        ahead = np.append(ahead, s_end)  # include the trajectory end (stop point / goal)
    targets = np.concatenate((behind, ahead))

    w = interp_fields(f, targets, v_min)
    w["t"] = w["t"] - w["t"][0]
    w["s"] = targets - targets[0]
    return w


def pack(w, n_max, spacing, seq, stamp_sec, stamp_nanosec, big_endian=False):
    order = ">" if big_endian else "<"
    n_valid = len(w["t"])
    if not 2 <= n_valid <= n_max:
        raise ValueError(f"n_valid {n_valid} outside 2..{n_max}")
    # pad with the last valid point so the packet size never changes
    idx = np.concatenate((np.arange(n_valid), np.full(n_max - n_valid, n_valid - 1)))

    block = np.empty((len(FIELDS), n_max), dtype=np.float32)
    for row, name in enumerate(FIELDS):
        block[row] = np.asarray(w[name], dtype=np.float64)[idx]
    yaw = block[FIELDS.index("yaw")]
    yaw[yaw.astype(np.float64) >= TWO_PI] = 0.0  # float32 rounding can turn 2pi - eps into 2pi

    spacing_mm = int(min(max(round(spacing * 1000.0), 0), 0xFFFF))
    header = struct.pack(order + HEADER_FMT, MAGIC, VERSION, n_max, n_valid, spacing_mm,
                         seq & 0xFFFFFFFF, stamp_sec, stamp_nanosec)
    return header + block.astype(order + "f4").tobytes()


def unpack(packet, n_max, big_endian=False):
    order = ">" if big_endian else "<"
    header = struct.unpack_from(order + HEADER_FMT, packet)
    data = np.frombuffer(packet, dtype=order + "f4",
                         offset=struct.calcsize(order + HEADER_FMT)).reshape(len(FIELDS), n_max)
    return header, data


def selftest(out_path=None):
    results = []

    def check(name, passed, detail=""):
        results.append(bool(passed))
        print(f"[{'PASS' if passed else 'FAIL'}] {name:30s} {detail}")

    n_max, nb, bs, v_min = 31, 6, 0.5, 0.1
    s = np.arange(0.0, 100.0 + 1e-9, 0.2)
    f = {"t": s / 10.0, "x": s.copy(), "y": np.zeros_like(s), "yaw": np.zeros_like(s),
         "v": np.full_like(s, 10.0), "a": np.zeros_like(s), "s": s.copy(), "kappa": np.zeros_like(s)}

    s_ego, offset = project_ego(f, 10.07, 0.3, 0.0)
    check("projection", abs(s_ego - 10.07) < 1e-9 and abs(offset - 0.3) < 1e-9,
          f"s_ego {s_ego:.4f} m, offset {offset:.3f} m")

    sp = choose_spacing(f, s_ego, n_max - nb - 1, 0.3, 3.0, 3.0)
    w = sample_window(f, s_ego, sp, n_max, nb, bs, v_min)
    check("window around the car",
          sp == 1.25 and len(w["t"]) == n_max and w["s"][0] == 0.0 and w["t"][0] == 0.0
          and np.allclose(w["s"][:nb + 1], bs * np.arange(nb + 1))
          and np.allclose(w["s"][nb:], nb * bs + sp * np.arange(n_max - nb))
          and np.allclose(w["t"][:nb + 1], 0.05 * np.arange(nb + 1))
          and abs(w["t"][-1] - w["t"][nb] - 3.0) < 1e-9 and abs(w["x"][nb] - 10.07) < 1e-9,
          f"spacing {sp} m, car at s {w['s'][nb]:.2f} m / t {w['t'][nb]:.2f} s, "
          f"horizon {w['t'][-1] - w['t'][nb]:.2f} s")

    slow, fast = dict(f), dict(f)
    slow["t"], fast["t"] = s / 1.0, s / 20.0
    sp_slow = choose_spacing(slow, s_ego, n_max - nb - 1, 0.3, 3.0, 3.0)
    sp_capped = choose_spacing(fast, s_ego, n_max - nb - 1, 0.3, 1.5, 3.0)
    check("spacing limits", sp_slow == 0.3 and sp_capped == 1.5,
          f"1 m/s: {sp_slow} m, 20 m/s capped: {sp_capped} m")

    s_start, _ = project_ego(f, 1.0, 0.0, 0.0)
    w0 = sample_window(f, s_start, 1.0, n_max, nb, bs, v_min)
    check("extrapolated behind start",
          abs(w0["x"][0] + 2.0) < 1e-9 and np.all(np.diff(w0["s"]) > 0) and np.all(np.diff(w0["t"]) > 0)
          and abs(w0["t"][nb] - 0.3) < 1e-9, f"first point x {w0['x'][0]:.2f} m (expected -2.00)")

    for big in (False, True):
        packet = pack(w, n_max, sp, 7, 123, 456, big)
        header, data = unpack(packet, n_max, big)
        same = all(np.array_equal(data[r], np.asarray(w[n], dtype=np.float32)) for r, n in enumerate(FIELDS))
        check(f"pack {'big' if big else 'little'} endian",
              len(packet) == packet_size(n_max) == 1016
              and header == (MAGIC, VERSION, n_max, n_max, 1250, 7, 123, 456) and same,
              f"{len(packet)} bytes")

    s_end, _ = project_ego(f, 95.0, 0.0, 0.0)
    header, data = unpack(pack(sample_window(f, s_end, 1.0, n_max, nb, bs, v_min), n_max, 1.0, 0, 0, 0), n_max)
    check("trajectory end + padding", header[3] == 12 and np.all(data[:, 12:] == data[:, 11:12]),
          f"n_valid {header[3]} (expected 12)")

    s_past, _ = project_ego(f, 105.0, 0.0, 0.0)
    w_past = sample_window(f, s_past, 1.0, n_max, nb, bs, v_min)
    check("car past the end", len(w_past["t"]) == nb + 1 and np.all(np.diff(w_past["s"]) > 0),
          f"n_valid {len(w_past['t'])} (expected {nb + 1})")

    g = dict(f)
    g["yaw"] = wrap_0_2pi(np.linspace(-0.1, 0.1, s.size))
    wy = sample_window(g, project_ego(g, 45.0, 0.0, 0.0)[0], 1.0, n_max, nb, bs, v_min)["yaw"]
    check("yaw across 0/2pi", np.all((wy >= 0.0) & (wy < TWO_PI))
          and np.minimum(wy, TWO_PI - wy).max() <= 0.1 + 1e-9,
          f"max distance to 0: {np.minimum(wy, TWO_PI - wy).max():.3f} rad")

    print(f"{sum(results)}/{len(results)} checks passed")
    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(pack(w, n_max, sp, 7, 123, 456))
        print(f"wrote {out_path}: version 3, seq 7, n_valid 31, spacing_mm 1250, car at point 7, "
              f"s 0..33 m (car at 3.0), t 0..3.3 s (car at 0.3), x 7.07..40.07 m, v 10 m/s")
    return all(results)


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        sys.exit(0 if selftest(sys.argv[i + 1] if i + 1 < len(sys.argv) else None) else 1)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from ioniq5_vehicle_launch.msg import ControllerTrajectory

    class AutoboxUdpSender(Node):
        def __init__(self):
            super().__init__("autobox_udp_sender")
            p = self.declare_parameter
            in_topic = p("input_topic", "/HI5/autoware_controller_trajectory").value
            odom_topic = p("odometry_topic", "/localization/kinematic_state").value
            self.target = (p("autobox_ip", "10.0.10.20").value, int(p("autobox_port", 44444).value))
            local_ip = p("local_ip", "").value
            self.local_port = int(p("local_port", 0).value)
            self.n_max = int(p("n_max", 31).value)
            self.n_behind = int(p("n_behind", 6).value)
            self.behind_spacing = float(p("behind_spacing_m", 0.5).value)
            self.min_spacing = float(p("min_spacing_m", 0.3).value)
            self.max_spacing = float(p("max_spacing_m", 3.0).value)
            self.min_horizon = float(p("min_horizon", 3.0).value)
            self.v_min = float(p("time_min_velocity", 0.1).value)
            self.max_offset = float(p("max_lateral_offset", 3.0).value)
            self.big_endian = bool(p("big_endian", False).value)
            max_packet = int(p("max_packet_bytes", 1032).value)

            size = packet_size(self.n_max)
            if size > max_packet:
                raise ValueError(f"packet {size} B exceeds {max_packet} B, reduce n_max")
            if not 1 <= self.n_behind <= self.n_max - 3:
                raise ValueError("need 1 <= n_behind <= n_max - 3")
            if not 0.0 < self.min_spacing <= self.max_spacing <= 65.535 or self.behind_spacing <= 0.0:
                raise ValueError("need 0 < min_spacing_m <= max_spacing_m <= 65.535 and behind_spacing_m > 0")
            if self.v_min <= 0.0:
                raise ValueError("time_min_velocity must be > 0")
            self.n_intervals = self.n_max - self.n_behind - 1

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if local_ip or self.local_port:
                self.sock.bind((local_ip, self.local_port))
            self.sock.setblocking(False)
            self.seq = 0
            self.ego = None
            self.ego_stamp = None

            qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
            self.create_subscription(Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)
            self.create_subscription(ControllerTrajectory, in_topic, self.on_trajectory, qos)
            # only needed when bound to a port the Autobox sends to; harmless otherwise
            self.create_timer(0.2, self.drain_feedback)
            self.get_logger().info(
                f"{in_topic} -> udp://{self.target[0]}:{self.target[1]} from port {self.local_port or 'auto'}, "
                f"{self.n_max} points: {self.n_behind} behind every {self.behind_spacing} m, car at index "
                f"{self.n_behind}, {self.n_intervals} ahead every {self.min_spacing}-{self.max_spacing} m "
                f"for >= {self.min_horizon} s, {size} bytes, {'big' if self.big_endian else 'little'} endian")

        def drain_feedback(self):
            while True:
                try:
                    self.sock.recvfrom(2048)
                except (BlockingIOError, InterruptedError):
                    return
                except OSError:
                    return

        def on_odom(self, msg):
            q = msg.pose.pose.orientation
            self.ego = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                        math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
            self.ego_stamp = msg.header.stamp

        def on_trajectory(self, msg):
            f = {name: np.asarray(getattr(msg, name), dtype=np.float64) for name in FIELDS}
            n = len(f["t"])
            if (n < 2 or any(len(val) != n for val in f.values())
                    or np.any(np.diff(f["s"]) <= 0.0) or np.any(np.diff(f["t"]) <= 0.0)):
                self.get_logger().warn("invalid trajectory, not sent", throttle_duration_sec=1.0)
                return
            if self.ego is None:
                self.get_logger().warn("no odometry yet, not sent", throttle_duration_sec=1.0)
                return

            s_ego, offset = project_ego(f, *self.ego)
            if offset > self.max_offset:
                self.get_logger().warn(f"car is {offset:.1f} m away from the trajectory", throttle_duration_sec=1.0)
            spacing = choose_spacing(f, s_ego, self.n_intervals, self.min_spacing, self.max_spacing, self.min_horizon)
            w = sample_window(f, s_ego, spacing, self.n_max, self.n_behind, self.behind_spacing, self.v_min)
            horizon = w["t"][-1] - w["t"][self.n_behind]
            if len(w["t"]) == self.n_max and horizon < self.min_horizon - 1e-3:
                self.get_logger().warn(
                    f"points ahead cover only {horizon:.1f} s (spacing capped at {self.max_spacing} m)",
                    throttle_duration_sec=2.0)

            packet = pack(w, self.n_max, spacing, self.seq,
                          self.ego_stamp.sec, self.ego_stamp.nanosec, self.big_endian)
            try:
                self.sock.sendto(packet, self.target)
            except (BlockingIOError, OSError) as e:
                self.get_logger().warn(f"UDP send failed: {e}", throttle_duration_sec=1.0)
                return
            self.seq = (self.seq + 1) & 0xFFFFFFFF

    rclpy.init()
    node = AutoboxUdpSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()