"""Compatibility import for the renamed G1BallMoveTeleop task."""

from theta_bench.tasks import g1_ball_move_teleop as _canonical

G1StickMoveTeleop = _canonical.G1BallMoveTeleop


def __getattr__(name: str):
    return getattr(_canonical, name)
