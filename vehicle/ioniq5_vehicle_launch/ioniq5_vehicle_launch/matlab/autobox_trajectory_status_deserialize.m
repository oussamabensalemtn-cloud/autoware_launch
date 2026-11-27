function [ok, s] = autobox_trajectory_status_deserialize(data)
%AUTOBOX_TRAJECTORY_STATUS_DESERIALIZE Decode the trajectory status packet (MRM PC -> MicroAutoBox, UDP 44445).
%
%   [ok, s] = autobox_trajectory_status_deserialize(data)
%
%   data  uint8 vector, one UDP datagram (80 bytes)
%   ok    true if size, magic (0x41575453 "AWTS") and version (1) match. If false, s holds zeros.
%   s     struct with the fields below. Floats that the PC does not have are NaN
%         (NaN compares false, e.g. traj_age_s <= 0.1 is false).
%
%   The status is sent at 50 Hz next to the trajectory stream on port 44444 and refers to the
%   last trajectory packet (traj_seq / traj_stamp are the seq and stamp of that packet).
%
%   offset type   field              meaning
%     0   uint32  magic              0x41575453
%     4   uint16  version            1
%     6   uint16  reserved
%     8   uint32  seq                status packet counter
%    12   int32   stamp_sec          send time (ROS time)          -> s.stamp [s]
%    16   uint32  stamp_nanosec
%    20   uint32  traj_seq           seq of the last trajectory packet on 44444
%    24   int32   traj_stamp_sec     stamp of that trajectory packet -> s.traj_stamp [s]
%    28   uint32  traj_stamp_nanosec
%    32   single  traj_age_s         now - planner stamp of that trajectory
%    36   single  since_last_traj_s  now - send time of that trajectory packet
%    40   uint8   valid              1 = trajectory packet recent (<= max age: 0.1 s MRM, 0.3 s standard),
%                                    plan recent, car <= 3 m from it and (MRM stack) /mrm/status not invalid
%    41   uint8   source             0 unknown, 1 standard stack (/planning/trajectory), 2 MRM stack
%    42   uint8   mrm_state          0 n/a, 1 standby, 2 active, 3 stopped, 4 invalid
%    43   uint8   mrm_level          /mrm/status level: 0 OK, 1 WARN, 2 ERROR, 3 STALE, 255 n/a
%    44   uint8   triggered          1 = MRM stack latched the trigger
%    45   uint8   n_valid            valid points in the last trajectory packet
%    46   uint16  warnings           bit 0 path_short, 1 plan_stop_infeasible, 2 object_blend_extended,
%                                    3 object_stop, 4 object_stop_infeasible, 5 no_objects, 6 stale_plan
%    48   single  horizon_s          planned time ahead of the car in the last packet
%    52   single  end_speed_mps      speed at the last valid point (0 = window ends in standstill)
%    56   single  ego_offset_m       car -> trajectory distance
%    60   single  heading_err_rad    car heading - trajectory heading at the car
%    64   single  odom_age_s         now - stamp of the odometry used for the last packet
%    68   single  plan_age_s         /mrm/status: age of Autoware's plan
%    72   single  stop_distance_m    /mrm/status: distance to the MRM stop point
%    76   single  path_remaining_m   /mrm/status: length of Autoware's path ahead
%
%   Decodes byte by byte, so the result does not depend on the endianness of the target.
%   Code generation compatible (Simulink MATLAB Function block, input uint8 [80x1] or [1x80]).
%   Python side: ioniq5_vehicle_launch/launch/autobox_udp_sender.py (pack_status).
%#codegen

s = struct( ...
    'seq', uint32(0), 'stamp', 0, 'traj_seq', uint32(0), 'traj_stamp', 0, ...
    'traj_age_s', single(0), 'since_last_traj_s', single(0), ...
    'valid', false, 'source', uint8(0), 'mrm_state', uint8(0), 'mrm_level', uint8(0), ...
    'triggered', false, 'n_valid', uint8(0), 'warnings', uint16(0), ...
    'horizon_s', single(0), 'end_speed_mps', single(0), 'ego_offset_m', single(0), ...
    'heading_err_rad', single(0), 'odom_age_s', single(0), 'plan_age_s', single(0), ...
    'stop_distance_m', single(0), 'path_remaining_m', single(0));
ok = false;

if numel(data) ~= 80
    return;
end
b = uint8(data(:));

if u32(b, 1) ~= uint32(1096242259) || u16(b, 5) ~= uint16(1)  % 0x41575453, version 1
    return;
end

ok = true;
s.seq = u32(b, 9);
s.stamp = double(i32(b, 13)) + double(u32(b, 17)) * 1e-9;
s.traj_seq = u32(b, 21);
s.traj_stamp = double(i32(b, 25)) + double(u32(b, 29)) * 1e-9;
s.traj_age_s = f32(b, 33);
s.since_last_traj_s = f32(b, 37);
s.valid = b(41) ~= 0;
s.source = b(42);
s.mrm_state = b(43);
s.mrm_level = b(44);
s.triggered = b(45) ~= 0;
s.n_valid = b(46);
s.warnings = u16(b, 47);
s.horizon_s = f32(b, 49);
s.end_speed_mps = f32(b, 53);
s.ego_offset_m = f32(b, 57);
s.heading_err_rad = f32(b, 61);
s.odom_age_s = f32(b, 65);
s.plan_age_s = f32(b, 69);
s.stop_distance_m = f32(b, 73);
s.path_remaining_m = f32(b, 77);
end

function v = u16(b, i)
v = bitor(uint16(b(i)), bitshift(uint16(b(i + 1)), 8));
end

function v = u32(b, i)
v = bitor(bitor(uint32(b(i)), bitshift(uint32(b(i + 1)), 8)), ...
          bitor(bitshift(uint32(b(i + 2)), 16), bitshift(uint32(b(i + 3)), 24)));
end

function v = i32(b, i)
v = typecast(u32(b, i), 'int32');
end

function v = f32(b, i)
v = typecast(u32(b, i), 'single');
end
