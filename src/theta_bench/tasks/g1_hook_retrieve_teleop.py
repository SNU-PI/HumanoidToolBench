"""Compatibility import for the renamed G1BallRetrieveTeleop task."""

from theta_bench.tasks import g1_ball_retrieve_teleop as _canonical

G1HookRetrieveTeleop = _canonical.G1BallRetrieveTeleop


def __getattr__(name: str):
    return getattr(_canonical, name)
