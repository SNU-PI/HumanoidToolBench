# Environments

HumanoidToolBench evaluates a Unitree G1 on 18 canonical conditions: three
scenarios, three levels and two tool-set modes. Every condition uses the same
budget: 100 episodes with seeds 10000 through 10099, and 3000 policy-control
steps per episode at 50 Hz (60 s of policy control, after the controller's
startup stabilization). Policies use the 36-value `decoupled_v1` action
contract described in the
[policy interface](PUBLIC_EVALUATION.md#policy-interface).

Environment IDs have the form `humanoidtoolbench/G1<Scenario>-L<level>-<mode>`.
`htb-eval` accepts them with or without the `humanoidtoolbench/`
prefix. `--list-envs` prints them in the order that `all` runs them: the nine
S conditions first, then the nine R conditions. The default condition is
`G1BallMove-L0-S`.

## The 18 conditions

| Environment ID | Scenario | Level | Mode | Instruction sent to the policy |
| --- | --- | --- | --- | --- |
| `G1BallMove-L0-S` | BallMove | L0 select | S | Pick the tool for moving the ball to the target. |
| `G1BallMove-L0-R` | BallMove | L0 select | R | Pick the tool for moving the ball to the target. |
| `G1BallMove-L1-S` | BallMove | L1 stationary use | S | Pick the tool and move the ball to the target. |
| `G1BallMove-L1-R` | BallMove | L1 stationary use | R | Pick the tool and move the ball to the target. |
| `G1BallMove-L2-S` | BallMove | L2 mobile use | S | Pick the tool and move the ball to the target. |
| `G1BallMove-L2-R` | BallMove | L2 mobile use | R | Pick the tool and move the ball to the target. |
| `G1BallRetrieve-L0-S` | BallRetrieve | L0 select | S | Pick the tool for retrieving the ball to the target. |
| `G1BallRetrieve-L0-R` | BallRetrieve | L0 select | R | Pick the tool for retrieving the ball to the target. |
| `G1BallRetrieve-L1-S` | BallRetrieve | L1 stationary use | S | Pick the tool and retrieve the ball to the target. |
| `G1BallRetrieve-L1-R` | BallRetrieve | L1 stationary use | R | Pick the tool and retrieve the ball to the target. |
| `G1BallRetrieve-L2-S` | BallRetrieve | L2 mobile use | S | Pick the tool and retrieve the ball to the target. |
| `G1BallRetrieve-L2-R` | BallRetrieve | L2 mobile use | R | Pick the tool and retrieve the ball to the target. |
| `G1IceBreak-L0-S` | IceBreak | L0 select | S | Pick the tool for breaking the ice blocks. |
| `G1IceBreak-L0-R` | IceBreak | L0 select | R | Pick the tool for breaking the ice blocks. |
| `G1IceBreak-L1-S` | IceBreak | L1 stationary use | S | Pick the tool and break the ice blocks. |
| `G1IceBreak-L1-R` | IceBreak | L1 stationary use | R | Pick the tool and break the ice blocks. |
| `G1IceBreak-L2-S` | IceBreak | L2 mobile use | S | Pick the tool and break the ice blocks. |
| `G1IceBreak-L2-R` | IceBreak | L2 mobile use | R | Pick the tool and break the ice blocks. |

The instruction depends only on the scenario and on whether the level is L0;
S and R conditions receive the same wording. The exact strings live in
[instructions.py](../src/humanoidtoolbench/instructions.py).

## Scenarios

The robot stands in front of a single 1.80 by 0.80 m bench top; the
placement code treats its left and right halves as separate benches, called
the left and right bench below. Tools always lie on the left bench. At L0 and L1 the task object starts on the right
bench; the Levels section describes what L2 moves to the left bench.

| Scenario | Reasoning axis | Correct tool | Confusable tool added in mode R | Task |
| --- | --- | --- | --- | --- |
| BallMove | Spatial: the ball is out of reach of the shorter tool | Long stick, 0.50 m | Short stick, 0.30 m, drawn from the same rod pool with the same cross-section and mass | Push a ball into a painted ring of radius 0.12 m. At L0 and L1 the ring centre is 0.22 to 0.32 m from the ball. |
| BallRetrieve | Affordance: only a hook can pull | Hook, 0.50 m | Straight stick of the same length and mass | Pull a ball back into a ring of radius 0.12 m. At L0 and L1 the ring is 0.20 m nearer the robot than the ball. |
| IceBreak | Physical: a stiff, heavy head delivers the impact that breaks ice | Metal hammer, 0.42 m, 0.15 kg | Light, compliant decoy (fly swatter, paint roller or plunger mesh), 0.04 kg | Break both 0.16 m ice cubes. |

Both modes also place two small irrelevant household objects (fruit,
tableware, containers, books, candles, clocks and the like) on the tool
bench. Object positions, tool slots and object identities are drawn from the
episode seed.

## Levels

| Level | What the policy must do | Layout |
| --- | --- | --- |
| L0 | Select the correct tool and lift it. | Object and goal on the right bench, as in L1, but only the pick is scored. |
| L1 | Select the tool and use it where the robot stands. | Object and goal on the right bench. |
| L2 | Select the tool and use it after moving. | The goal (BallMove ring), the object (BallRetrieve ball) or the second block (IceBreak) is placed on the far strip of the left bench, so the robot has to step along the bench. |

## Modes

| Mode | Tool bench contents | Paper notation |
| --- | --- | --- |
| S | Correct tool and two irrelevant objects (3 objects) | R0 |
| R | Same as S plus one confusable tool from the scenario's tool set (4 objects) | R1 |

The difference between R and S measures the cost of ruling out a plausible
wrong tool. The instruction is identical, and the irrelevant objects are
drawn from the same pool in the same way.

## Success criteria

Success is checked every control step and must hold for 50 consecutive steps
(1.0 s). An episode ends on that 50th step or when the step budget runs out;
there is no failure termination, so a wrong pick or a dropped ball only costs
time.

| Level | Predicate | Values |
| --- | --- | --- |
| L0 | The correct tool's body has risen at least 80 % of the lift height above its height at the first check. | Lift height 0.10 m, so at least 0.08 m; held 50 steps. |
| L1 and L2, BallMove | The ball centre is inside the ring. | Ring radius 0.12 m. |
| L1 and L2, BallRetrieve | The ball centre is inside the target ring. | Ring radius 0.12 m. |
| L1 and L2, IceBreak | Both blocks are broken and their shards have scattered. | A block breaks when the normal contact force from a tool, sampled once per 50 Hz control step and multiplied by the 0.005 s physics step, reaches 0.042 N s while that tool was closing at 0.5 m/s or faster; shards must spread at least 0.03 m. |

At L1 and L2 the pick itself is not part of the predicate; picking, touching
and locomotion are recorded as metrics. The L2 predicate is the same as L1;
the layout forces the walk. Any tool body can deliver the IceBreak impact;
the compliant decoy normally cannot reach the threshold.

## Per-episode outputs

Each run directory contains `summaries/episode_<seed>.json` with `success`,
`step`, `outcome` (`success` or `timeout` for canonical tasks),
`final_metrics`, `ever_true` and `first_true_step` per metric, plus a
per-step `metrics/episode_<seed>.jsonl` and four videos under
`videos/episode_<seed>/`. Metric names:

| Metric | Meaning |
| --- | --- |
| `tool_lift`, `picked_correct_tool` | Height gained by the correct tool in metres, and whether it is at least 0.08 m at this step. |
| `first_touch_role`, `first_touch_correct`, `wrong_touch_count` | Which object a hand touched first (0 correct tool, 1 confusable tool, 2 irrelevant object, -1 none yet) and how many wrong objects were touched before the correct tool. |
| `tool_touched_target`, `wrong_tool_touched_target` | Whether the correct or the confusable tool has contacted the ball or ice. |
| `on_spot`, `push_distance`, `reach_shortfall` (BallMove) | Ball inside the ring at this step; metres pushed from the start; a diagnostic of how far the ball starts beyond arm reach plus the short stick. |
| `in_target_area`, `retrieve_gain` (BallRetrieve) | Ball inside the ring at this step; metres the ball has come towards the robot's working spot (the spawn at L0 and L1, the standing spot at L2). |
| `ice_broken`, `ice_broken_count`, `ice_broke_1`, `ice_broke_2`, `peak_impulse` (IceBreak) | Both blocks broken; number broken; at least one broken; both broken; largest qualifying impact in N s. A block counts as broken once it has fractured and its shards have spread at least 0.03 m. |
| `dist_to_target_bench`, `reached_target_bench`, `first_move_toward_target` (L2) | Distance from the pelvis to the standing spot in metres; latched within 0.40 m; whether the first 0.15 m of movement went towards the spot. |
| `level`, `reasoning_mode` | 0, 1 or 2; 0 for S and 1 for R. |

`ever_true` can be true for `on_spot`, `in_target_area` or `ice_broken` in an
episode whose `success` is false when the outcome did not hold for 50 steps.

The condition-level score is `successes / episodes` in `benchmark_result.json`,
written only after the outcome and video checks described in
[PUBLIC_EVALUATION.md](PUBLIC_EVALUATION.md#results) pass.
