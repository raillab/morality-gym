from morality_gym.environments.core.action import SubActionEnum
from morality_gym.environments.core.event import Event


def action_norm_fn(event: Event):
    is_player_action = event.is_action and len(event.initiated_entities) == 1 and event.initiated_entities[0].is_player
    if is_player_action and event.sub_action == SubActionEnum.PUSH:
        return {"push"}
    else:
        return {}