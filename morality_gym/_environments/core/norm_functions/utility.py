from morality_gym._environments.core.event import Event

######################
# -    UTILITY     - #
# - NORM FUNCTIONS - #
######################

def n_character_harm_fn(event: Event):
    # if len(affected_entities) != 1:
    #     raise ValueError(f"utility_fn not supported for len(affected_entities) = {len(affected_entities)}")

    utilities = {}
    for entity in event.affected_entities:
        if entity.character_type is not None:
        # if "character_type" in entity.traits:
            curr_utility = f"n_{entity.character_type}_harm"
            if curr_utility not in utilities:
                utilities[curr_utility] = entity.amount
            else:
                utilities[curr_utility] += entity.amount

    return utilities

def n_trait_harmed_fn(event: Event):
    raise NotImplementedError


REGISTRY = {
    "n_character_harmed": n_character_harm_fn,
}

def make_fn(name: str, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"norm_fn '{name}' not found in REGISTRY")

    def _fn(event: Event):
        return REGISTRY[name](event, **kwargs)

    return _fn