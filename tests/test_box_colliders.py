"""Fitted box colliders for the meshes listed in assets.tools.BOX_COLLIDER_UIDS."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from humanoidtoolbench.assets import tools
from humanoidtoolbench.engines.mujoco import MujocoSimulator


def _write_obj(path, points) -> str:
    path.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in points))
    return str(path)


def _corners(pos, quat, half) -> np.ndarray:
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return np.asarray(pos) + (signs * np.asarray(half)) @ rot.reshape(3, 3).T


def test_axis_aligned_box_mesh_is_fitted_exactly(tmp_path) -> None:
    half = np.array([0.25, 0.01, 0.02])
    centre = np.array([0.1, -0.2, 0.3])
    signs = np.array(list(np.ndindex(2, 2, 2))) * 2 - 1
    points = centre + half * signs
    path = _write_obj(tmp_path / "box.obj", points)
    pos, quat, fitted = MujocoSimulator._box_collider(path, None)
    np.testing.assert_allclose(sorted(fitted), sorted(half), atol=1e-9)
    np.testing.assert_allclose(pos, centre, atol=1e-9)


def test_fitted_box_bounds_every_scaled_vertex(tmp_path) -> None:
    rng = np.random.default_rng(0)
    # A thin rod with a near-square section, turned off the axes: the case a
    # principal-axis box gets diagonal.
    rod = rng.uniform([-0.25, -0.006, -0.005], [0.25, 0.006, 0.005], size=(64, 3))
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, np.array([np.cos(0.3), 0.2, np.sin(0.3), 0.1]) / np.linalg.norm([np.cos(0.3), 0.2, np.sin(0.3), 0.1]))
    points = rod @ turn.reshape(3, 3).T
    scale = [-1.2, 0.9, 1.1]
    path = _write_obj(tmp_path / "rod.obj", points)
    pos, quat, half = MujocoSimulator._box_collider(path, scale)
    scaled = points * np.asarray(scale)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    local = (scaled - np.asarray(pos)) @ rot.reshape(3, 3)
    assert np.all(np.abs(local) <= np.asarray(half) + 1e-9)
    # Tight in its own axes, and no bigger than the box along the file axes,
    # which is one of the candidates the fit searches.
    extent = np.ptp(scaled @ rot.reshape(3, 3), axis=0)
    np.testing.assert_allclose(2 * np.asarray(half), np.maximum(extent, 1e-3), atol=1e-9)
    assert np.prod(2 * np.asarray(half)) <= np.prod(np.ptp(scaled, axis=0)) + 1e-12


def test_listed_meshes_are_flagged_and_others_are_not() -> None:
    assert len(tools.BOX_COLLIDER_UIDS) == 10
    assert all(len(uid) == 32 for uid in tools.BOX_COLLIDER_UIDS)
    try:
        listed = tools.mesh_asset("long_stick", uid="0eedc541d1bb4a32935fc6ed7a2fea08", length=0.5)
        other = tools.mesh_asset("long_stick", uid="cb84d2732c9840498598e9838b368385", length=0.5)
    except tools.MissingToolAssets:
        pytest.skip("tool meshes are not installed")
    assert listed.box_colliders is True
    assert other.box_colliders is False
