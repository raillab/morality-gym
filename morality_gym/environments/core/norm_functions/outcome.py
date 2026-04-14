from typing import Callable, Optional, Dict, Set

from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.event import Event

######################
# -    OUTCOME     - #
# - NORM FUNCTIONS - #
######################
def outcome_fn(
        event: Event,
        filter_fn: Callable[[Event, BaseEntity], bool],
        name_fn: Callable[[Event, BaseEntity], str]
):
    norms = set()
    for entity in event.affected_entities:
        if not filter_fn(event, entity):
            curr_norm = name_fn(event, entity)
            norms.add(curr_norm)
    return norms

# def outcome_fn_alt(
#         event: Event,
#         filter_fn: Optional[Callable[[Event, BaseEntity], bool]] = None,
#         include_amount: bool = False,
#         include_character_type: bool = True,
#         include_traits: Optional[Dict[str, bool]] = None,
# ):
#     def _name_fn(_event: Event, _entity: BaseEntity):
#         outcome_descr=_event.outcome_descr
#         properties = []
#         if include_amount:
#             properties.append(f"amount={_entity.amount}")
#         if include_character_type:
#             properties.append(f"character_type={_entity.character_type}")
#         if include_traits is not None:
#             for trait, is_include in include_traits.items():
#                 if is_include:
#                     properties.append(f"traits.{trait}={_entity.traits[trait]}")
#
#     if filter_fn is None:
#         filter_fn = lambda _event, _entity: False

# def harm_fn_alt(
#         event: Event,
#         name_fn: Optional[Callable[[Event], str]] = None,
#         filter_fn: Optional[Callable[[Event], bool]] = None
# ):
#     norms = set()
#     if name_fn is None:
#         name_fn = lambda _entity: _entity.name
#
#     for entity in event.affected_entities:
#         if entity.character_type is not None:
#             curr_norm = f"{entity.character_type}_harm"
#             norms.add(curr_norm)
#     return norms
######################

#
# REGISTRY = {
#     "harm": harm_fn,
# }

def make_fn(
        # filter_fn: Optional[Callable[[Event, BaseEntity], bool]] = None,
        include_outcome_descr: Optional[Set[str]] = None,  # Useful??
        include_character_types: Optional[Set[str]] = None,

        name_include_amount: bool = False,
        name_include_character_type: bool = True,
        name_include_traits: Optional[Dict[str, bool]] = None,
        name_include_is_player: bool = False,
        name_include_subsets: bool = False
) -> Callable[[Event], Set[str]]:

    if name_include_subsets:
        raise NotImplementedError

    def _name_fn(_event: Event, _entity: BaseEntity):
        outcome_descr=_event.outcome_descr
        properties = []
        if name_include_amount:
            properties.append(f"amount={_entity.amount}")
        if name_include_character_type:
            properties.append(f"character_type={_entity.character_type}")
        if name_include_is_player:
            properties.append(f"is_player={_entity.is_player}")
        if name_include_traits is not None:
            # NOTE: Assumes traits are properly formatted
            for trait, is_include in name_include_traits.items():
                if is_include:
                    properties.append(f"traits.{trait}={_entity.traits[trait]}")

        properties.sort()

        _name = f"{outcome_descr}:{';'.join(properties)}"
        return _name

    def _filter_fn(_event: Event, _entity: BaseEntity):
        # is_filter = False
        if include_character_types is not None:
            if _entity.character_type not in include_character_types:
                return True

        if include_outcome_descr is not None:
            if _event.outcome_descr not in include_outcome_descr:
                return True

        return False

    def _outcome_fn(_event: Event):
        return outcome_fn(event=_event, filter_fn=_filter_fn, name_fn=_name_fn)

    return _outcome_fn
