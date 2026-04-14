from morality_gym.environments.core.dynamics.base import BaseDynamics
from morality_gym.environments.core.event import Event
from morality_gym.environments.core.state import WorldState


class ProcessEventsDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState
    ):
        super().__init__(world_state)

    @staticmethod
    def _compute_named_causal(event: Event) -> Event:
        action_ancestors = []
        # Recursively find ancestors of event that have action_descr not equal to None - i.e. events that caused event
        def recursive_find_ancestors(curr_event):
            for child in curr_event.prev_events:
                if not child.is_pseudo_event:  # Ignore pseudo events
                    if child.action_descr is not None:
                        action_ancestors.append(child)
                    else:
                        recursive_find_ancestors(child)

        init_entities = []
        for ancestor in action_ancestors:
            init_entities.extend(ancestor.initiated_entities)

        recursive_find_ancestors(event)

        action_descr = list(sorted([ancestor.action_descr for ancestor in action_ancestors]))
        causal_sub_descr = "_".join(action_descr) + "_cause_" + event.outcome_descr
        causal_descr = f"cause_{event.outcome_descr}"

        causal_event = Event(
            timestep=event.timestep,
            is_causal=True,
            causal_descr=causal_descr,
            causal_sub_descr=causal_sub_descr,
            next_events=[event],
            prev_events=action_ancestors,
            initiated_entities=init_entities,
            affected_entities=event.affected_entities,
            is_pseudo_event=True
        )

        # NOTE: This may cause issues - so check if is_pseudo when working with events
        for prev_event in action_ancestors:
            prev_event.next_events.append(causal_event)
        event.prev_events.append(causal_event)

        return causal_event

    def _compute_all_named_causal(self):
        named_recent_outcomes = [event for event in self._world_state.recent_outcome_events if
                                 event.outcome_descr is not None]
        for outcome in named_recent_outcomes:
            causal_event = self._compute_named_causal(outcome)
            self._world_state.add_event(causal_event, event_type="causal", is_proc_causal=True)

    def __call__(self):
        self._compute_all_named_causal()
        # raise NotImplementedError