###########
# OUTCOME #
###########
from morality_gym.environments.core.event import Event


def outcome_harm_fn(event: Event):
    norms = set()
    for entity in event.affected_entities:
        if entity.character_type is not None:
            # if "character_type" in entity.traits:
            curr_norm = f"{entity.character_type}_harm"
            norms.add(curr_norm)
    return norms

###########

###########
# UTILITY #
###########
def utility_character_harm_fn(event: Event):
    # if len(affected_entities) != 1:
    #     raise ValueError(f"utility_fn not supported for len(affected_entities) = {len(affected_entities)}")

    utilities = {}
    for entity in event.affected_entities:
        if entity.character_type is not None:
        # if "character_type" in entity.traits:
            curr_utility = f"n_{entity.character_type}_harmed"
            if curr_utility not in utilities:
                utilities[curr_utility] = entity.amount
            else:
                utilities[curr_utility] += entity.amount

    return utilities
###########
