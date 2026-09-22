"""Read a recorded LeRobot episode and its ToolBench environment state."""

import json


def get_episode_lerobot(dataset, eps_idx, data_format=None):
    del data_format

    def _to_int(value):
        item = getattr(value, "item", None)
        if callable(item):
            return int(item())
        return int(value)

    from_idx = _to_int(dataset.episode_data_index["from"][eps_idx])
    to_idx = _to_int(dataset.episode_data_index["to"][eps_idx])
    episode = [dataset[i] for i in range(from_idx, to_idx)]

    try:
        encoded = dataset.meta.episodes[eps_idx]["environment_config"]
    except KeyError as exc:
        raise KeyError(f"episode {eps_idx} has no environment_config") from exc
    env_conf = json.loads(encoded)
    if not isinstance(env_conf, dict):
        raise TypeError(f"episode {eps_idx} environment_config must be a mapping")
    return env_conf, episode
