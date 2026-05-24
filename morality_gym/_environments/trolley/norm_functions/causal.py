from morality_gym._environments.core.event import Event


def cause_harm_norm_fn(event: Event):
    norms = set()
    if event.is_causal and event.causal_descr == "cause_harm":
        is_cause_personal = False
        for prev_event in event.prev_events:
            if prev_event.action_descr == "personal":
                is_cause_personal = True
                break
        if is_cause_personal:
            norm_prefix = "personal_action_caused"
        else:
            norm_prefix = "impersonal_action_caused"

        norms.add(f"{norm_prefix}_harm")
        for entity in event.affected_entities:
            if entity.character_type is not None:
                norms.add(f"{norm_prefix}_{entity.character_type}_harm")
                if entity.is_player:
                    norms.add(f"{norm_prefix}_agent_harm")

    return norms


def means_to_end_norm_fn():
    pass


def harm_prevented_norm_fn():
    pass