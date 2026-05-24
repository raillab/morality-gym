from __future__ import annotations

import copy
from enum import IntEnum
from typing import Tuple, Optional, Set, Dict, List, Callable, Union, Any

import numpy as np

from morality_gym._environments.core.action import ActionEnum, SubActionEnum
from morality_gym._environments.core.custom_types import PosType, StateChangeType
from morality_gym._environments.core.event import Event


###############
# BASE ENTITY #
###############
class BaseEntity:
    def __init__(
            self,
            name: str,
            group: str,
            pos: PosType,

            is_collidable: bool,
            is_movable: bool,

            is_actable: bool = False,
            is_interactable: bool = False,
            is_agent_interactable: bool = False,
            is_scripted: bool = False,
            is_harmable: bool = False,
            is_intersectable: bool = False,
            is_terminatable: bool = False,
            is_static: bool = False,

            pos_bounds: Optional[Tuple[PosType, PosType]] = None,

            amount: int = 1,
            valid_amounts: Optional[List[int]] = None,
            traits: Optional[Dict[str, Any]] = None,

            # First StateChangeType corresponds to initiating entity and second corresponds to affected entity (self)
            interact_fn: Optional[Callable[[BaseEntity, BaseEntity, Optional[np.random.Generator], ...],
                                            Tuple[StateChangeType, StateChangeType, bool]]] = None,
            sub_interact_fns: Optional[Dict[SubActionEnum,
                                            Callable[[BaseEntity, BaseEntity, Optional[np.random.Generator], ...],
                                                      Tuple[StateChangeType, StateChangeType, bool]]]] = None,

            interact_type: Optional[SubActionEnum] = None,
            interact_descr: Optional[str] = None,

            prob_harm_collide: float = 1.0,
            traversability_group: int = 0,

            to_asset_fn: Optional[Callable[[BaseEntity], Union[str, List[str]]]] = None,
            to_rot_fn: Optional[Callable[[BaseEntity], Union[int, List[int]]]] = None,
            to_alpha_fn: Optional[Callable[[BaseEntity], Union[int, List[int]]]] = None,

            vis_layer: int = 1,
            is_vis_amount: bool = False,

            character_type: Optional[str] = None,
            valid_character_types: Optional[List[str]] = None,

            has_post_step: bool = None,
            has_pre_step: bool = None
    ):
        self._name = name
        self._group = group
        self.pos = pos
        self._pos_bounds = pos_bounds
        self.next_pos = None

        self.character_type = character_type
        self.valid_character_types = valid_character_types

        self.amount = amount
        if valid_amounts is None:
            valid_amounts = [amount]
        self.valid_amounts = valid_amounts
        if min(self.valid_amounts) == 0:
            raise ValueError("0 is not a valid amount in valid_amounts arg. Must only include values >= 1.")

        if traits is None:
            traits = {}
        self.traits = traits

        self._is_player = False

        self._init_pos = pos
        self._init_is_collidable = is_collidable
        self._init_is_movable = is_movable
        self._init_is_actable = is_actable
        self._init_is_interactable = is_interactable
        self._init_is_agent_interactable = is_agent_interactable
        self._init_is_harmable = is_harmable
        self._init_is_intersectable = is_intersectable

        self.is_collidable = is_collidable
        self.is_movable = is_movable
        self.is_actable = is_actable
        self.is_interactable = is_interactable
        self.is_agent_interactable = is_agent_interactable
        self.is_harmable = is_harmable
        self.is_intersectable = is_intersectable
        self.is_static = is_static

        self.is_scripted = is_scripted
        self.is_terminatable = is_terminatable

        #########################
        # --- PRE/POST STEP --- #
        #########################
        if has_post_step is None:
            has_post_step = False
        if has_pre_step is None:
            if is_collidable:
                has_pre_step = True
            else:
                has_pre_step = False

        # If this entity has post step or pre step functions
        self.has_post_step = has_post_step
        self.has_pre_step = has_pre_step

        self.pre_step_pos = None
        #########################

        self.action: Optional[ActionEnum] = None
        self.action_taken: Optional[ActionEnum] = None
        self.sub_action_taken: Optional[SubActionEnum] = None

        ############
        # INTERACT #
        ############
        self.interact_fn = interact_fn
        self.interact_type = interact_type
        self.interact_descr = interact_descr

        # if sub_interact_fns is not None:
        #     raise NotImplementedError("sub_interact_fns arg not yet supported.")
        if sub_interact_fns is None:
            sub_interact_fns = {}

        def _push_interact(
                initiating_entity: BaseEntity,
                affected_entity: BaseEntity,
                rng: Optional[np.random.Generator] = None,
                **_kwargs
        ) -> Tuple[StateChangeType, StateChangeType, bool]:
            # y1, x1 = initiating_entity.pos
            # y2, x2 = self.pos

            y1, x1 = initiating_entity.pos
            y2, x2 = affected_entity.pos

            prev_action_taken = copy.copy(affected_entity.action_taken)

            if y1 == y2:
                if x1 < x2:
                    # Push Right
                    affected_entity.action_taken = ActionEnum.RIGHT
                else:
                    # Push LEFT
                    affected_entity.action_taken = ActionEnum.LEFT
            elif x1 == x2:
                if y1 < y2:
                    # Push DOWN
                    affected_entity.action_taken = ActionEnum.DOWN
                else:
                    # Push UP
                    affected_entity.action_taken = ActionEnum.UP
            else:
                raise ValueError(
                    f"Invalid positions for self and interact_entity when pushing: {affected_entity.pos}, {initiating_entity.pos}")

            state_change_init = {}
            state_change_affect = {
                "action_taken": (prev_action_taken, affected_entity.action_taken),
            }

            return state_change_init, state_change_affect, False
        sub_interact_fns[SubActionEnum.PUSH] = _push_interact


        self.sub_interact_fns = sub_interact_fns
        ############

        if not 0 <= traversability_group <= 2:
            raise ValueError(f"traversability_group does not satisfy 0 <= traversability_group <= 2. "
                             f"traversability_group={traversability_group}")
        self._traversability_group = traversability_group

        #####################
        # - RENDERER VARS - #
        #####################
        self.to_asset_fn = to_asset_fn
        self.to_rot_fn = to_rot_fn
        self.to_alpha_fn = to_alpha_fn
        self.vis_layer = vis_layer
        self.is_vis_amount = is_vis_amount
        #####################

        self.is_terminated = False
        self.termination_reason: Optional[str] = None

        # Not sure if useful - may remove
        self.just_collided = False
        self.just_moved = False
        self.just_harmed = False

        self.prob_harm_collide = prob_harm_collide
        # Curr harm
        self.curr_harm = HarmEnum.NONE

        self.events: Dict[str, List[Event]] = {}

    #################
    # MAGIC METHODS #
    #################
    def __hash__(self):
        # name assumed to be unique
        return self._name
    #################

    ##############
    # PROPERTIES #
    ##############
    @property
    def name(self) -> str:
        return self._name

    @property
    def group(self) -> str:
        return self._group

    @property
    def is_harmed(self):
        return self.curr_harm != HarmEnum.NONE

    @property
    def traversability_group(self) -> int:
        return self._traversability_group

    @property
    def is_player(self) -> bool:
        return self._is_player
    ##############

    ###############
    # SET METHODS #
    ###############
    def set_action(
            self,
            action: ActionEnum,
            next_pos: Optional[Tuple[int, int]] = None,
            sub_action: Optional[SubActionEnum] = None,
    ):
        self.action = action
        self.action_taken = action
        self.sub_action_taken = sub_action
        if action == ActionEnum.MOVE_TO_POS:
            self.next_pos = next_pos

    def set_state(
            self,
            **states
    ):
        if "pos" in states:
            self.pos = states["pos"]

    ###############

    ################
    # - INTERACT - #
    ################
    # def _interact_default(
    #         self,
    #         val: Optional[Any] = None,
    #         interact_entity: Optional[BaseEntity] = None,
    #         prev_event: Optional[Event] = None
    # ):
    #     pass
    #
    # def _interact_push(
    #         self,
    #         interact_entity: Optional[BaseEntity],
    #         prev_event: Optional[Event] = None
    # ):
    #     # interact_entity is entity pushing this entity
    #
    #     if not self.is_actable:
    #         raise NotImplementedError("Pushing entity with is_actable=False is currently not supported")
    #
    #     y1, x1 = interact_entity.pos
    #     y2, x2 = self.pos
    #
    #     prev_action_taken = copy.copy(self.action_taken)
    #
    #     if y1 == y2:
    #         if x1 < x2:
    #             # Push Right
    #             self.action_taken = ActionEnum.RIGHT
    #         else:
    #             # Push LEFT
    #             self.action_taken = ActionEnum.LEFT
    #     elif x1 == x2:
    #         if y1 < y2:
    #             # Push DOWN
    #             self.action_taken = ActionEnum.DOWN
    #         else:
    #             # Push UP
    #             self.action_taken = ActionEnum.UP
    #     else:
    #         raise ValueError(
    #             f"Invalid positions for self and interact_entity when pushing: {self.pos}, {interact_entity.pos}")
    #
    #     if prev_action_taken != self.action_taken:
    #         # Create event
    #         event = Event(
    #             host_entity=self,
    #             prev_events=[prev_event],
    #             action_type="direct", action=ActionEnum.INTERACT, sub_action=SubActionEnum.PUSH,
    #             state_change={
    #                 "action_taken": (prev_action_taken, self.action_taken),
    #             }
    #         )
    #         if prev_event is not None:
    #             prev_event.next_events.append(event)
    #         self.events["action_taken"] = [event]
    #
    # def _interact_pickup_dropoff(
    #         self,
    #         val: Optional[Any] = None,
    #         interact_entity: Optional[BaseEntity] = None
    # ):
    #     pass
    #
    # def interact(
    #         self,
    #         val: Optional[Any] = None,
    #         interact_entity: Optional[BaseEntity] = None,
    #         interact_type: SubActionEnum = SubActionEnum.DEFAULT,
    #         prev_event: Optional[Event] = None
    # ):
    #     if interact_type == SubActionEnum.DEFAULT:
    #         self._interact_default(val, interact_entity, prev_event)
    #     elif interact_type == SubActionEnum.PUSH:
    #         if interact_entity is None:
    #             raise ValueError("interact_entity must be specified for SubActionEnum.PUSH.")
    #         self._interact_push(interact_entity=interact_entity, prev_event=prev_event)
    #     elif interact_type == SubActionEnum.PICKUP_DROPOFF:
    #         self._interact_pickup_dropoff(val, interact_entity)
    #     else:
    #         raise ValueError(f"interact_type={interact_type} not supported.")
    ################

    # TODO: Test
    def reset_to_init(self, excludes: Optional[Set[str]] = None):
        if excludes is None or len(excludes) == 0:
            self.pos = self._init_pos
            self.next_pos = None
            self.is_collidable = self._init_is_collidable
            self.is_movable = self._init_is_movable
            self.is_actable = self._init_is_actable
            self.is_interactable = self._init_is_interactable
            self.is_agent_interactable = self._init_is_agent_interactable
            self.is_harmable = self._init_is_harmable
            self.is_intersectable = self._init_is_intersectable

            self.curr_harm = HarmEnum.NONE
            self.is_terminated = False
            self.termination_reason = None
            # self.is_harmed = False
            self.events = {}
        else:
            raise NotImplementedError

    def set_terminated(self, reason: str):
        self.is_terminated = True

        self.is_movable = False
        self.is_actable = False
        # Note: Not setting is_harmable since it may still be harmed if it is terminated

        self.termination_reason = reason

    def set_harm(
            self,
            harm: HarmEnum,
            # reason: str
    ):
        if not self.is_harmable:
            raise ValueError("Cannot set harm since entity is not harmable.")

        self.curr_harm = harm
        self.just_harmed = True
        if harm == HarmEnum.MAJOR:
            self.set_terminated("Major Harm")
            # print(f"{self.name} terminated due to major harm.")
            # Question: Should I modify is_harmable?

    def script(self):
        raise NotImplementedError

    def calc_obs(self, is_normalise: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def calc_pos_obs(self, is_normalise: bool = False) -> Union[Tuple[float, float], PosType]:
        y, x = self.pos
        if is_normalise:
            (min_y, min_x), (max_y, max_x) = self._pos_bounds

            norm_y = (y - min_y) / (max_y - min_y)
            norm_x = (x - min_x) / (max_x - min_x)
            pos = (norm_y, norm_x)
        else:
            pos = self.pos

        return pos

    def post_step(self):
        self.action = None
        self.action_taken = None
        # if self.is_harmable and self.curr_harm == HarmEnum.MAJOR:
        #     self.is_terminated = True
        #     self.termination_reason = "Major Harm"
        #     print(f"{self.name} terminated due to major harm.")

    def pre_step(self):
        self.just_collided = False
        self.just_moved = False
        self.just_harmed = False

        self.sub_action_taken = None  # How to handle?

        self.pre_step_pos = self.pos


class HarmEnum(IntEnum):
    NONE = 0
    MINOR = 1
    MAJOR = 2