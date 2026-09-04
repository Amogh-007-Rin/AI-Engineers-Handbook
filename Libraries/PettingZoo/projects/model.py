"""Dependency-light PettingZoo parallel-step contract validator."""


def validate_transition(active_agents, actions, transition):
    if set(actions) != set(active_agents):
        raise ValueError("one action per active parallel agent required")
    observations, rewards, terminations, truncations, infos = transition
    expected = set(active_agents)
    for name, mapping in (("rewards", rewards), ("terminations", terminations),
                          ("truncations", truncations), ("infos", infos)):
        if set(mapping) != expected:
            raise ValueError(f"{name} keys must match active agents")
    if not set(observations).issubset(expected):
        raise ValueError("unknown observation agent")
    return True
