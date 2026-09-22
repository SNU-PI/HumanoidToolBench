"""Controller-neutral action contracts used by THETA-Bench policies."""

from .g1 import (
    DECOUPLED_ACTION_DIM,
    DECOUPLED_ACTION_ORDER,
    DECOUPLED_ACTION_SLICES,
    DECOUPLED_SCHEMA_ID,
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_HAND_JOINTS,
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_HAND_JOINTS,
    SONIC_LATENT_ACTION_DIM,
    SONIC_LATENT_ACTION_ORDER,
    SONIC_LATENT_ACTION_SLICES,
    SONIC_LATENT_SCHEMA_ID,
    DecoupledActionCodec,
    DecoupledGoal,
    SonicLatentActionCodec,
    SonicLatentGoal,
)

__all__ = [
    "DECOUPLED_ACTION_DIM",
    "DECOUPLED_ACTION_ORDER",
    "DECOUPLED_ACTION_SLICES",
    "DECOUPLED_SCHEMA_ID",
    "G1_LEFT_ARM_JOINTS",
    "G1_LEFT_HAND_JOINTS",
    "G1_RIGHT_ARM_JOINTS",
    "G1_RIGHT_HAND_JOINTS",
    "SONIC_LATENT_ACTION_DIM",
    "SONIC_LATENT_ACTION_ORDER",
    "SONIC_LATENT_ACTION_SLICES",
    "SONIC_LATENT_SCHEMA_ID",
    "DecoupledActionCodec",
    "DecoupledGoal",
    "SonicLatentActionCodec",
    "SonicLatentGoal",
]
