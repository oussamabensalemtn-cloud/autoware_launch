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


# Trajectory status packet (MRM PC -> MicroAutoBox, default port 44445), little endian, 80 bytes, sent at 50 Hz.
# Tells the tracking controller whether the trajectory stream on 44444 is up to date and valid.
# Floats that are not available are NaN. The MATLAB deserializer lives in the MABX Simulink model.
STATUS_MAGIC = 0x41575453  # "AWTS"
STATUS_VERSION = 1
STATUS_FIELDS = (
    ("magic", "I"), ("version", "H"), ("reserved", "H"), ("seq", "I"), ("stamp_sec", "i"), ("stamp_nanosec", "I"),
    ("traj_seq", "I"), ("traj_stamp_sec", "i"), ("traj_stamp_nanosec", "I"),  # seq / stamp of the last packet on 44444
    ("traj_age_s", "f"),         # now - stamp of the planner trajectory in the last packet
    ("since_last_traj_s", "f"),  # now - send time of the last trajectory packet
    ("valid", "B"),              # 1 = the last trajectory packet is fresh and usable (see status_valid)
    ("source", "B"),             # 0 unknown, 1 /planning/trajectory (standard stack), 2 /mrm/trajectory (MRM stack)
    ("mrm_state", "B"),          # 0 n/a, 1 standby, 2 active, 3 stopped, 4 invalid (from /mrm/status)
    ("mrm_level", "B"),          # /mrm/status level: 0 OK, 1 WARN, 2 ERROR, 3 STALE, 255 n/a
    ("triggered", "B"),          # 1 = the MRM stack has latched the trigger
    ("n_valid", "B"),            # valid points in the last trajectory packet
    ("warnings", "H"),           # bit field, see MRM_WARNINGS
    ("horizon_s", "f"),          # planned time covered ahead of the car in the last packet
    ("end_speed_mps", "f"),      # speed at the last valid point of the last packet (0 = window ends in standstill)
    ("ego_offset_m", "f"),       # distance car -> trajectory at the time of the last packet
    ("heading_err_rad", "f"),    # car heading - trajectory heading at the car, (-pi, pi]
    ("odom_age_s", "f"),         # now - stamp of the odometry used for the last packet
    ("plan_age_s", "f"),         # /mrm/status: age of Autoware's plan used by the MRM
    ("stop_distance_m", "f"),    # /mrm/status: distance to the MRM stop point
    ("path_remaining_m", "f"),   # /mrm/status: length of Autoware's path ahead
)
STATUS_FMT = "<" + "".join(t for _, t in STATUS_FIELDS)
STATUS_SIZE = struct.calcsize(STATUS_FMT)  # 80
MRM_STATES = {"standby": 1, "active": 2, "stopped": 3, "invalid": 4}
MRM_WARNINGS = ("path_short", "plan_stop_infeasible", "object_blend_extended", "object_stop",
                "object_stop_infeasible", "no_objects", "stale_plan")  # bit 0, 1, ...


def warning_bits(text):
    names = {w.strip() for w in text.split(",") if w.strip()}
    return sum(1 << i for i, w in enumerate(MRM_WARNINGS) if w in names)


def pack_status(**values):
    """Fields of STATUS_FIELDS as keyword arguments; magic, version and reserved are set here."""
    values.update(magic=STATUS_MAGIC, version=STATUS_VERSION, reserved=0)
    masks = {"B": 0xFF, "H": 0xFFFF, "I": 0xFFFFFFFF}
    out = []
    for name, t in STATUS_FIELDS:
        v = values[name]
        if t == "f":
            out.append(float(v))
        elif t in masks:
            out.append(int(v) & masks[t])
        else:
            out.append(int(v))
    return struct.pack(STATUS_FMT, *out)


def unpack_status(packet):
    """dict of the fields, or None for a wrong size, magic or version."""
    if len(packet) != STATUS_SIZE:
        return None
    d = dict(zip((n for n, _ in STATUS_FIELDS), struct.unpack(STATUS_FMT, packet)))
    return d if d["magic"] == STATUS_MAGIC and d["version"] == STATUS_VERSION else None


def status_valid(since_last, traj_age, offset, max_age, max_offset, mrm_required, mrm_state):
    """valid flag of the status packet: a recent packet, a recent plan, the car near it and, in the MRM stack,
    /mrm/status present and not invalid."""
    ok = (since_last is not None and since_last <= max_age and traj_age is not None and traj_age <= max_age
          and offset is not None and offset <= max_offset)
    if mrm_required:
        ok = ok and mrm_state in (1, 2, 3)
    return bool(ok)


def heading_at(f, s_ego):
    return float(np.interp(s_ego, f["s"], np.unwrap(f["yaw"])))


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


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


def selftest(out_path=None, status_path=None):
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

    sample = dict(seq=5, stamp_sec=123, stamp_nanosec=500000000, traj_seq=7, traj_stamp_sec=123,
                  traj_stamp_nanosec=456, traj_age_s=0.03, since_last_traj_s=0.012, valid=1, source=2,
                  mrm_state=2, mrm_level=1, triggered=1, n_valid=31, warnings=warning_bits("path_short, no_objects"),
                  horizon_s=3.3, end_speed_mps=0.0, ego_offset_m=0.02, heading_err_rad=-0.01, odom_age_s=0.008,
                  plan_age_s=0.06, stop_distance_m=25.5, path_remaining_m=float("nan"))
    st = unpack_status(pack_status(**sample))
    same = st is not None and all(
        (math.isnan(st[k]) and math.isnan(v)) if isinstance(v, float) and math.isnan(v)
        else abs(st[k] - v) < 1e-6 for k, v in sample.items())
    check("status packet", STATUS_SIZE == 80 and same and st["warnings"] == 0b100001
          and unpack_status(pack_status(**sample)[:-1]) is None
          and unpack_status(b"\0" + pack_status(**sample)[1:]) is None, f"{STATUS_SIZE} bytes")
    check("status valid flag",
          status_valid(0.02, 0.05, 0.1, 0.1, 3.0, True, 1) and not status_valid(0.02, 0.05, 0.1, 0.1, 3.0, True, 4)
          and not status_valid(0.2, 0.05, 0.1, 0.1, 3.0, False, 0) and status_valid(0.02, 0.05, 0.1, 0.1, 3.0, False, 0)
          and not status_valid(None, None, None, 0.1, 3.0, False, 0) and not status_valid(0.02, 0.05, 4.0, 0.1, 3.0, False, 0),
          "fresh/stale/invalid MRM/off the trajectory")

    print(f"{sum(results)}/{len(results)} checks passed")
    if status_path:
        with open(status_path, "wb") as fh:
            fh.write(pack_status(**sample))
        print(f"wrote {status_path}: {sample}")
    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(pack(w, n_max, sp, 7, 123, 456))
        print(f"wrote {out_path}: version 3, seq 7, n_valid 31, spacing_mm 1250, car at point 7, "
              f"s 0..33 m (car at 3.0), t 0..3.3 s (car at 0.3), x 7.07..40.07 m, v 10 m/s")
    return all(results)


def main():
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        files = sys.argv[i + 1:i + 3]
        sys.exit(0 if selftest(*files) else 1)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from diagnostic_msgs.msg import DiagnosticStatus
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
            self.status_target = (self.target[0], int(p("status_port", 44445).value))
            status_rate = float(p("status_rate_hz", 50.0).value)
            self.source = int(p("trajectory_source", 0).value)  # 1 standard stack, 2 MRM stack
            self.max_age = float(p("status_max_age_s", 0.1).value)
            mrm_status_topic = p("mrm_status_topic", "/mrm/status").value
            self.mrm_status_timeout = float(p("mrm_status_timeout_s", 0.5).value)

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
            self.status_seq = 0
            self.last = None        # what went out in the last trajectory packet
            self.mrm = None         # last /mrm/status
            self.mrm_time = None

            qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
            self.create_subscription(Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)
            self.create_subscription(ControllerTrajectory, in_topic, self.on_trajectory, qos)
            # only needed when bound to a port the Autobox sends to; harmless otherwise
            self.create_timer(0.2, self.drain_feedback)
            if self.status_target[1] > 0 and status_rate > 0.0:
                self.create_subscription(DiagnosticStatus, mrm_status_topic, self.on_mrm_status, 10)
                self.create_timer(1.0 / status_rate, self.send_status)
                self.get_logger().info(
                    f"trajectory status -> udp://{self.status_target[0]}:{self.status_target[1]} at "
                    f"{status_rate:.0f} Hz (source {self.source}, max age {self.max_age} s)")
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
            n_valid = len(w["t"])
            self.last = {
                "seq": self.seq, "stamp": (self.ego_stamp.sec, self.ego_stamp.nanosec),
                "sent": self.get_clock().now().nanoseconds * 1e-9,
                "planner": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "odom": self.ego_stamp.sec + self.ego_stamp.nanosec * 1e-9,
                "n_valid": n_valid, "horizon": float(horizon), "end_speed": float(w["v"][n_valid - 1]),
                "offset": offset, "heading_err": wrap_pi(self.ego[2] - heading_at(f, s_ego)),
            }
            self.seq = (self.seq + 1) & 0xFFFFFFFF

        def on_mrm_status(self, msg):
            self.mrm = msg
            self.mrm_time = self.get_clock().now().nanoseconds * 1e-9

        def send_status(self):
            now_msg = self.get_clock().now().to_msg()
            now = now_msg.sec + now_msg.nanosec * 1e-9
            nan = float("nan")
            last = self.last
            v = dict(seq=self.status_seq, stamp_sec=now_msg.sec, stamp_nanosec=now_msg.nanosec, traj_seq=0,
                     traj_stamp_sec=0, traj_stamp_nanosec=0, traj_age_s=nan, since_last_traj_s=nan, valid=0,
                     source=self.source, mrm_state=0, mrm_level=255, triggered=0, n_valid=0, warnings=0,
                     horizon_s=nan, end_speed_mps=nan, ego_offset_m=nan, heading_err_rad=nan, odom_age_s=nan,
                     plan_age_s=nan, stop_distance_m=nan, path_remaining_m=nan)
            if last is not None:
                v.update(traj_seq=last["seq"], traj_stamp_sec=last["stamp"][0], traj_stamp_nanosec=last["stamp"][1],
                         traj_age_s=now - last["planner"], since_last_traj_s=now - last["sent"],
                         n_valid=last["n_valid"], horizon_s=last["horizon"], end_speed_mps=last["end_speed"],
                         ego_offset_m=last["offset"], heading_err_rad=last["heading_err"],
                         odom_age_s=now - last["odom"])
            if self.mrm is not None and now - self.mrm_time <= self.mrm_status_timeout:
                kv = {k.key: k.value for k in self.mrm.values}
                level = self.mrm.level  # 'byte' field: bytes of length 1 in rclpy
                level = level[0] if isinstance(level, (bytes, bytearray)) else int(level)
                v.update(mrm_state=MRM_STATES.get(kv.get("state", ""), 0), mrm_level=level,
                         triggered=int(kv.get("triggered", "") == "True"),
                         warnings=warning_bits(kv.get("warnings", "")))
                for field, key in (("plan_age_s", "plan_age_s"), ("stop_distance_m", "stop_distance_m"),
                                   ("path_remaining_m", "path_remaining_m")):
                    try:
                        v[field] = float(kv[key])
                    except (KeyError, ValueError):
                        pass
            v["valid"] = int(status_valid(
                v["since_last_traj_s"] if last else None, v["traj_age_s"] if last else None,
                v["ego_offset_m"] if last else None, self.max_age, self.max_offset, self.source == 2, v["mrm_state"]))
            try:
                self.sock.sendto(pack_status(**v), self.status_target)
            except (BlockingIOError, OSError) as e:
                self.get_logger().warn(f"UDP status send failed: {e}", throttle_duration_sec=1.0)
                return
            self.status_seq = (self.status_seq + 1) & 0xFFFFFFFF

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