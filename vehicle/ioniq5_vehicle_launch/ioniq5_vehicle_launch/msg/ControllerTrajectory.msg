# One array entry per trajectory point; all arrays have the same length as the autoware trajectory. SI units.
# header.stamp and header.frame_id (map) are copied from the Autoware trajectory.

std_msgs/Header header

float64[] t      # [s]     time of the point relative to header.stamp  0 at the first point
float64[] x      # [m]     position in header.frame_id
float64[] y      # [m]     position in header.frame_id
float64[] yaw    # [rad]   vehicle heading in [0, 2*pi), 0 = +x (east), counter-clockwise positive
float64[] v      # [m/s]   longitudinal velocity, negative when reversing
float64[] a      # [m/s^2] longitudinal acceleration
float64[] s      # [m]     travelled distance along the trajectory, 0 at the first point
float64[] kappa  # [1/m]   geometric curvature in point order, positive = turning left