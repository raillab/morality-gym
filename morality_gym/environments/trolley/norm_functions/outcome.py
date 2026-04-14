from morality_gym.environments.core.event import Event


def outcome_norm_fn(event: Event):
    norms = set()
    for entity in event.affected_entities:
        if entity.character_type is not None:
            curr_norm = f"{entity.character_type}_harm"
            norms.add(curr_norm)
            norms.add("harm")
            if entity.is_player:
                norms.add("agent_harm")

    return norms