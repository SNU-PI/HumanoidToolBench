# HumanoidToolBench data

**[ToolBook (snupilab/humanoidtoolbench-teleop)](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**,
the HumanoidToolBench demonstration dataset on Hugging Face, contains
simulation teleoperation data and real Unitree G1 recordings. The
**[HumanoidToolBench collection](https://huggingface.co/collections/snupilab/humanoidtoolbench-6aa6d654e29d415b0dafbe61)**
links this dataset and the model releases.

## Access and source revisions

The dataset page is public; downloading recordings requires an access request
that is approved manually. Request access on the dataset page, then
authenticate your downloader after approval. The dataset card limits use to
non-commercial research, disallows recording redistribution, and requests
citation. These dataset conditions are separate from the code's MIT license.

The collection links these source snapshots:

| Source | Pinned data location |
| --- | --- |
| Simulation | [raw/, revision 8b2cd31e107b64cb13f812ea217a63a20845c78a](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop/tree/8b2cd31e107b64cb13f812ea217a63a20845c78a/raw) |
| Real G1 | [hardware/, revision 47eca9322bb53fa1c685363271a87d2e414cb0e8](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop/tree/47eca9322bb53fa1c685363271a87d2e414cb0e8/hardware) |

Pin the dataset revision and selected episode manifest in your experiments.
The collection's simulation training count describes a curated set of segments;
it is not the size of the entire dataset repository or an instruction to train
on every uploaded episode. Inspect success labels, source episodes, and the
selected model's data recipe before defining a training split.

## Repository layout

The dataset currently exposes `raw/`, `hardware/`, `trimmed/`, `rendered/`,
`rendered_images/`, and `viewer/` directories. Their roles differ:

| Directory | Contents |
| --- | --- |
| `raw/` | Simulation recordings grouped by operator, environment, and domain-randomization level |
| `hardware/` | Real G1 recordings in the recorded `l1_stick_standard`, `l1_stick_reason`, `l1_hook_standard`, and `l1_hook_reason` groups |
| `trimmed/` | Slices of recorded trajectories, with source provenance |
| `rendered/`, `rendered_images/` | Additional rendering products; consult their release metadata |
| `viewer/` | Episode-level indexes for browsing, rather than frame-level action data |

Some releases under `rendered/`, such as `rendered/isaac/`, were produced with
an internal rendering pipeline that is not included in this toolkit.

Simulation recordings use LeRobot files with per-frame Parquet data, videos,
and metadata such as `meta/info.json`, `meta/episodes.jsonl`, and
`meta/simple_contract.json`. Read each recording's metadata to determine its
frame rate, image dimensions, recorded cameras, environment, and outcome.
Historical source names can differ from current canonical environment names:
`G1StickMove` is now `G1BallMove` and `G1HookRetrieve` is now `G1BallRetrieve`
(task UIDs `g1_stick_move_teleop` and `g1_hook_retrieve_teleop` likewise), and
recorded IDs may carry a `-v0` suffix. Keep the original paths as provenance
and use versionless IDs for new runs.

A recording's `meta/simple_contract.json` (`simple-lerobot-contract-v1`)
states its frame conventions. Each row is (observation at t, action at t,
outcome at t+1): `observation.*` is the state the action was chosen from,
`action` and `teleop.*` are the command issued at that step, and `next.*` and
`metric.*` are measured after it, so an episode's terminal observation is not
recorded. `next.success` and the episode's `episode_success` in
`meta/episodes.jsonl` are what the task reported, never a threshold on the
reward. Quaternions are scalar-first `[w, x, y, z]`, lengths are in metres
and joint values in radians.

Treat slices and rerendered views of the same trajectory as related samples
when splitting data. A render-validation result concerns the generated images;
it does not establish task success or make a second independent demonstration.

## State and action contracts

The public simulation evaluator sends a **32-value policy state** and accepts
**36-value `decoupled_v1` actions**. These are different from the raw recorded
**43-value `observation.state`** containing the robot's joint positions.

Simulation preprocessing derives the 32-value state from the hands, arms,
torso, and preceding base-height command. The first base-height value is 0.74.
The 36-value action contains hand and arm targets, torso orientation, base
height, and navigation commands. Use the exact order documented in the
[policy interface](../docs/PUBLIC_EVALUATION.md#policy-interface); passing the
raw 43-value observation directly changes the model input contract.

Real G1 recordings have their own recording/control contract. In particular,
the documented hardware adapter has 35 supervised action values and no
recorded absolute navigation target yaw. Executed waist targets also differ
from simulation torso commands. A hardware-trained action head therefore needs
a matching adapter; dimension padding alone does not establish compatibility.

## Local directories

Evaluation outputs default to `data/evals/`. Simulator meshes fetched by the
installer live separately under `data/ms_assets/`; these are runtime assets,
not the teleoperation dataset. Keep downloaded recordings and generated
evaluation outputs out of the code release, and preserve their source revision
and run metadata alongside them.
