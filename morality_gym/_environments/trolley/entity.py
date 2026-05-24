from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Callable, Set

import numpy as np

from morality_gym._environments.core.action import ActionEnum
from morality_gym._environments.core.custom_types import PosType, StateChangeType
from morality_gym._environments.core.entity.base import BaseEntity
from morality_gym._environments.core.entity.group import EntityGroup
from morality_gym._environments.core.event import Event


##########
#  RAIL  #
##########
class Rail(BaseEntity):
    def __init__(
            self,
            name: str,
            group: str,
            pos: Tuple[int, int],
            direction: Optional[str],  # One of ["L", "R", "U", "D"] or None
            prev_rails: Optional[List[Rail]] = None,
            next_rail: Optional[Rail] = None,
            _interact_fn: Optional[Callable[..., Tuple[StateChangeType, StateChangeType, bool]]] = None,
            _is_interactable: bool = False,
            _is_agent_interactable: bool = False,
            _is_static: bool = True

    ):
        def to_asset_fn(entity: BaseEntity) -> str:
            if not isinstance(entity, Rail):
                raise TypeError(f"Expected entity to be of type Rail, got {type(entity)}")

            is_curved = entity.calc_is_curved()

            if is_curved:
                return "rail_curved"
            else:
                return "rail_straight"

        def to_rot_fn(entity: BaseEntity) -> int:
            # Note: This needs optimisation
            is_curved = self.calc_is_curved()

            if not is_curved:
                if self.direction in {"L", "R"}:
                    return 90
                elif self.direction is None and self.next_rail is not None:
                    if self.pos[1] == self.next_rail.pos[1]:
                        return 0
                    else:
                        return 90
                else:
                    return 0
            else:

                yp, xp = self.prev_rails[0].pos  # prev
                yc, xc = self.pos
                yn, xn = self.next_rail.pos  # next
                # U -> R & R -> U
                if (xp == xn - 1 and yp == yn -1 and xp == xc and yn == yc) or (
                    xn == xp - 1 and yn == yp - 1 and xn == xc and yp == yc
                ):
                    return 90
                # U -> L & L -> U
                elif (xp == xn + 1 and yp == yn - 1 and xp == xc and yn == yc) or (
                        xn == xp + 1 and yn == yp - 1 and xn == xc and yp == yc
                ):
                    return 180
                # L -> D & D -> L
                elif (xp == xn - 1 and yp == yn - 1 and xn == xc and yp == yc) or (
                        xn == xp - 1 and yn == yp - 1 and xp == xc and yn == yc
                ):
                    return 270
                # R -> D & D -> R
                elif (xp == xn + 1 and yp == yn - 1 and xn == xc and yp == yc) or (
                        xn == xp + 1 and yn == yp - 1 and xp == xc and yn == yc
                ):
                    return 0
                else:
                    raise ValueError

                # return 270

                # U -> R, R -> U
                # U -> L, L -> U
                # D -> R, R -> D
                # D -> L, R -> L
                pass

        def to_alpha_fn(entity: BaseEntity) -> int:
            return 255

        super().__init__(
            name, group, pos, is_collidable=False, is_movable=False, is_actable=False,
            is_interactable=_is_interactable, is_agent_interactable=_is_agent_interactable,
            is_static=_is_static,
            interact_fn=_interact_fn,
            traversability_group=0, vis_layer=0,
            to_asset_fn=to_asset_fn, to_rot_fn=to_rot_fn, to_alpha_fn=to_alpha_fn
        )
        # super().__init__(
        #     name, group, is_collidable=False, is_movable=False, is_actable=False, is_interactable=False, pos=pos)

        if prev_rails is None:
            prev_rails = []

        self.prev_rails = prev_rails
        self.next_rail = next_rail
        self.direction = direction

        # self.to_asset_fn = to_asset_fn
        # self.to_rot_fn = to_rot_fn
        # self.to_alpha_fn = to_alpha_fn


    def __repr__(self):
        return f"Rail ({self._name})"

    def __str__(self):
        return self.__repr__()

    def calc_is_curved(self):
        # Check if rail is curved based off prev first rail and next rail - use only first element in
        # prev_rails
        if self.next_rail is not None and len(self.prev_rails) > 0:
            next_rail = self.next_rail
            prev_rail = self.prev_rails[0]
            if next_rail.pos[0] == prev_rail.pos[0] or next_rail.pos[1] == prev_rail.pos[1]:
                is_curved = False
            else:
                is_curved = True

        else:
            is_curved = False

        return is_curved

    def get_direction(self):
        return self.direction

    @staticmethod
    def calc_rel_direction(rail_1, rail_2):
        pos1 = rail_1.pos
        pos2 = rail_2.pos
        if pos1[0] < pos2[0]:
            rel_dir = "D"
        elif pos1[0] > pos2[0]:
            rel_dir = "U"
        elif pos1[1] < pos2[1]:
            rel_dir = "R"
        elif pos1[1] > pos2[1]:
            rel_dir = "L"
        else:
            raise ValueError("Invalid positions")
        return rel_dir

#################
#  RAIL SWITCH  #
#################
class RailSwitch(Rail):
    def __init__(
            self,
            name: str,
            group: str,
            pos: Tuple[int, int],
            prev_rails: Optional[List[Rail]] = None,
            connect_rails: Optional[List[Rail]] = None

    ):
        if connect_rails is None:
            self._connect_rails = None
            self._n_connect_rails = 0
            next_rail = None
            self._next_rail_ind = None
        else:
            self._connect_rails = connect_rails
            self._n_connect_rails = len(connect_rails)
            self._next_rail_ind = 0
            next_rail = self._connect_rails[self._next_rail_ind]

        def _interact_fn(
                initiating_entity: BaseEntity,
                affected_entity: RailSwitch,
                rng: Optional[np.random.Generator] = None,
                state: Optional[int] = None
                # **_kwargs
        ) -> Tuple[StateChangeType, StateChangeType, bool]:
            # state has to be declared as optional due to typing stuff
            if state is None:
                raise ValueError

            prev_rail_ind = affected_entity._next_rail_ind
            prev_next_rail = affected_entity.next_rail

            if 0 <= state < self._n_connect_rails:
                affected_entity._next_rail_ind = state
                affected_entity.next_rail = self._connect_rails[state]
                # TODO: Use rng for randomness if required
            else:
                raise ValueError(f"Invalid state = {state} in interact_fn for RailSwitch")

            state_change_init = {}
            state_change_affect = {
                "next_rail_ind": (prev_rail_ind, affected_entity._next_rail_ind),
                "next_rail": (prev_next_rail, affected_entity.next_rail)
            }
            return state_change_init, state_change_affect, False

        super().__init__(
            name=name,
            group=group,
            pos=pos, direction=None,
            prev_rails=prev_rails, next_rail=next_rail,
            _is_interactable=True, _is_agent_interactable=False,
            _interact_fn=_interact_fn, _is_static=False
        )

        # self.is_interactable = True
        # self._init_is_interactable = self.is_interactable
        # self.is_agent_interactable = False
        # self._init_is_agent_interactable = self.is_agent_interactable


        # Orientations corresponding to each connect_rail
        # NEW
        rail_orients = []
        if self._connect_rails is not None:
            for i, next_rail in enumerate(self._connect_rails):
                rail_orients.append(self.calc_rel_direction(self, next_rail))
        self._rail_orients = rail_orients

        # def to_rot_fn(entity: BaseEntity) -> int:
        #     if self.rail_orients is None:
        #         # TODO: Stuff here
        #         return 0
        #     curr_orient = self.rail_orients[self._next_rail_ind]
        #     if curr_orient[1] in {"L", "R"}:
        #         return 90
        #     else:
        #         return 0
        #
        # self.to_rot_fn = to_rot_fn


    @property
    def connect_rails(self):
        return self._connect_rails

    @property
    def n_connect_rails(self):
        return self._n_connect_rails

    def get_direction(self):
        if self._rail_orients is None:
            return None
        else:
            return self._rail_orients[self._next_rail_ind]

    def set_connect_rails(self, connect_rails: List[Rail]):
        if not 0 < len(connect_rails) <= 3:
            raise ValueError(f"Invalid number of connect rails = {len(connect_rails)}")
        self._connect_rails = connect_rails
        self._n_connect_rails = len(connect_rails)
        self._next_rail_ind = 0
        self.next_rail = self._connect_rails[self._next_rail_ind]

        rail_orients = []
        if self._connect_rails is not None:
            for i, next_rail in enumerate(self._connect_rails):
                rail_orients.append(self.calc_rel_direction(self, next_rail))
        self._rail_orients = rail_orients

    def set_state(
            self,
            **states
    ):
        if "next_rail_ind" in states:
            self._next_rail_ind = states["next_rail_ind"]
            self.next_rail = self._connect_rails[self._next_rail_ind]

        super().set_state(**states)

    def toggle_rail(self):
        self._next_rail_ind = (self._next_rail_ind + 1) % self._n_connect_rails
        self.next_rail = self._connect_rails[self._next_rail_ind]
        return self.next_rail

    # def _interact_default(
    #         self,
    #         val: Optional[Any] = None,
    #         interact_entity: Optional[BaseEntity] = None,
    #         prev_event: Optional[Event] = None
    # ):
    #     self._next_rail_ind = val
    #     self.next_rail = self._connect_rails[self._next_rail_ind]

    # def comp_rail_orients(self):
    #     # return
    #     assert len(self.prev_rails) == 1, "prev_rails must be of length 1"
    #     # Note: Relative directions
    #     prev_rail_dir = self.calc_rel_direction(self, self.prev_rails[0])
    #     connect_rail_dirs = [self.calc_rel_direction(self, connect_rail) for connect_rail in self._connect_rails]
    #     self._rail_orients = [(prev_rail_dir, curr_connect_dir) for curr_connect_dir in connect_rail_dirs]
    #     return self._rail_orients
        # print(self.rail_orients)

    def calc_obs(self) -> Dict[str, Any]:
        obs = {
            "next_rail": self._next_rail_ind
        }

        return obs
    # # Get rail orientation based off in rail and out rail
    # def get_rail_orient(self):
    #     pass




#################
# -- RAILWAY -- #
#################
# Restrictions
class Railway(EntityGroup):
    def __init__(
            self,
            name: str,
            group: str,
            # pos: PosType,
            schema: np.ndarray,  # coordinates in this will be relative to pos
            switches: Dict[PosType, Tuple[str, Tuple[str, ...]]],
            trolley_starts: List[PosType],  # TODO: Remove this - it will be set later
            trolley_start_mode: str = "landmark_reached",
            n_trolleys: int = 1,
            is_trolleys_terminatable: bool = True,
            is_trolleys_intersectable: bool = True,
            pos_bounds: Optional[Tuple[PosType, PosType]] = None,
    ):
        super().__init__(
            name,
            group,
            (0, 0),
            is_scripted=True,
            has_post_step=True,
            pos_bounds=pos_bounds
        )
        self._n_trolleys = n_trolleys

        self._rails: Dict[Tuple[int, int], Rail] = {}
        # self._rail_switches: Dict[Tuple[int, int], Rail] = {}
        self._switches: Dict[PosType, RailSwitch] = {}  # Note: switches is a subset of rails
        self._name_to_switch: Dict[str, RailSwitch] = {}

        self._starts: Dict[Tuple[int, int], Rail] = {}
        self._ends: Dict[Tuple[int, int], Rail] = {}

        self._schema = schema
        # Convert switches to absolute positions
        # abs_switches = {}
        # for rel_switch_pos, switch_tup in switches.items():
        #     abs_connect_arr = []
        #     for curr_rel_pos in switch_tup[1]:
        #         curr_abs_pos = (self.pos[0] + curr_rel_pos[0], self.pos[1] + curr_rel_pos[1])
        #         abs_connect_arr.append(curr_abs_pos)
        #     abs_switch_pos = (self.pos[0] + rel_switch_pos[0], self.pos[1] + rel_switch_pos[1])
        #     abs_switches[abs_switch_pos] = (switch_tup[0], tuple(abs_connect_arr))
        # switches = abs_switches

        self._build_from_schema(schema, switches)

        # CREATE TROLLEYS #

        # self._trolley_starts = [(trolley_pos[0] + pos[0], trolley_pos[1] + pos[1]) for trolley_pos in trolley_starts]
        if trolley_start_mode not in {"landmark_reached", "episode_start"}:
            raise ValueError(f"Invalid trolley_start_mode = {trolley_start_mode}. Supported values = ['landmark_reached', 'episode_start']")

        trolley_starts_enabled = trolley_start_mode == "episode_start"
        self._trolley_starts = trolley_starts
        self._trolleys: List[Trolley] = []  # TODO
        for i in range(n_trolleys):
            trolley = Trolley(
                name=f"trolley_{i}", group=f"{self.group}_trolley", curr_rail=self._rails[self._trolley_starts[i]],
                is_terminatable=is_trolleys_terminatable, is_intersectable=is_trolleys_intersectable,
                pos_bounds=pos_bounds, starts_enabled=trolley_starts_enabled,
            )
            self._trolleys.append(trolley)

        self._trolley_start_mode = trolley_start_mode

        for trolley in self._trolleys:
            self._entities[trolley.name] = trolley
        for rail in self._rails.values():
            self._entities[rail.name] = rail

    @staticmethod
    def calc_next_pos(
            pos: Tuple[int, int],
            direction: str,
            # bounds: Tuple[Tuple[int, int], Tuple[int, int]]
    ):
        # next_pos = pos
        if direction == "L":
            next_pos = (pos[0], pos[1] - 1)
        elif direction == "R":
            next_pos = (pos[0], pos[1] + 1)
        elif direction == "U":
            next_pos = (pos[0] - 1, pos[1])
        elif direction == "D":
            next_pos = (pos[0] + 1, pos[1])
        else:
            raise ValueError("Invalid direction")

        return next_pos
        # if (bounds[0][0] <= next_pos[0] < bounds[1][0]) and (bounds[0][1] <= next_pos[1] < bounds[1][1]) :
        #     return next_pos
        # else:
        #     return None

    # def _build_from_schema(
    #         self,
    #         schema: np.ndarray, switches: Dict[PosType, Tuple[str, Tuple[PosType, ...]]]
    # ):
    #     pass


    def _build_from_schema(
            self,
            schema: np.ndarray,
            switches: Dict[PosType, Tuple[str, Tuple[str, ...]]]
    ):
        rail_group = f"{self.group}_rail"
        switch_group = f"{self.group}_switch"

        # --- STEP 1 --- #
        # Extract coords of rails & switches
        #  - Extract coords and values of all elements that are not equal to X
        # Format of each tup is ((y,x),val)
        y_coords, x_coords = np.where(schema != 'X')
        values = schema[y_coords, x_coords]
        rails_d = {rel_pos: rail_val for (rel_pos, rail_val) in zip(zip(y_coords, x_coords), values)}
        # Update coords to be absolute
        rails_d = {(rel_pos[0] + self.pos[0], rel_pos[1] + self.pos[1]): rail_val for rel_pos, rail_val in
                   rails_d.items()}

        # --- STEP 2 --- #
        self._rails = {}
        self._switches = {}  # Note: switches is subset of rails
        self._name_to_switch = {}

        # Get position adjacent to pos based off direction i.e. if direction="up" then get position above pos
        def get_adj_pos(pos, direction):
            if direction == "L":
                adj_pos = (pos[0], pos[1] - 1)
            elif direction == "R":
                adj_pos = (pos[0], pos[1] + 1)
            elif direction == "U":
                adj_pos = (pos[0] - 1, pos[1])
            elif direction == "D":
                adj_pos = (pos[0] + 1, pos[1])
            else:
                raise ValueError("Invalid direction")
            return adj_pos

        # --- STEP 2a --- #
        # First pass to create all rails & switches
        for rail_pos, val in rails_d.items():
            if val == "S":
                if rail_pos not in switches:
                    raise ValueError(f"Switch at {rail_pos} not found in switches")

                name = switches[rail_pos][0]
                rail_switch = RailSwitch(name, switch_group, rail_pos)
                self._rails[rail_pos] = rail_switch
                self._switches[rail_pos] = rail_switch
                self._name_to_switch[name] = rail_switch

            elif val in {"L", "R", "U", "D"}:
                name = f"{self.name}_{str(rail_pos)}"
                rail = Rail(name, rail_group, rail_pos, val)
                self._rails[rail_pos] = rail
            else:
                raise ValueError(f"Invalid value = {val} in schema at pos={rail_pos}")

        # --- STEP 2b --- #
        # Second pass to set prev_rails and next_rail for all rails and switches
        # & connect_rails for all switches
        for rail_pos, rail in self._rails.items():
            if isinstance(rail, RailSwitch):
                connect_rails = []
                for connect_dir in switches[rail_pos][1]:
                    next_pos = self.calc_next_pos(rail_pos, connect_dir)
                    if next_pos in self._rails:
                        next_rail = self._rails[next_pos]
                        next_rail.prev_rails.append(rail)
                        connect_rails.append(next_rail)
                    else:
                        raise ValueError(f"Direction={rail.direction} for connection in rail switch at pos={rail_pos} "
                                         f"leads to invalid position")

                rail.set_connect_rails(connect_rails)
            else:
                next_pos = self.calc_next_pos(rail_pos, rail.direction)
                if next_pos in self._rails:
                    next_rail = self._rails[next_pos]
                    rail.next_rail = next_rail
                    next_rail.prev_rails.append(rail)
                else:
                    rail.next_rail = None

        # --- STEP 3 --- #
        for rail_pos, rail in self._rails.items():
            if rail.next_rail is None:
                self._ends[rail_pos] = rail
            if len(rail.prev_rails) == 0:
                self._starts[rail_pos] = rail

        # for switch in self._switches.values():
        #     # TODO: Come back to this :)
        #     switch.comp_rail_orients()

    def toggle_switch(self, name):
        return self._name_to_switch[name].toggle_rail()

    @property
    def switches(self):
        # NOTE: Returns name_to_switch
        return self._name_to_switch

    def script(self):
        # raise NotImplementedError
        for trolley in self._trolleys:
            if trolley.is_actable:
                curr_rail = self._rails[trolley.pos]
                if curr_rail.next_rail is not None:
                    if trolley.is_enabled:
                        trolley.set_action(ActionEnum.MOVE_TO_POS, curr_rail.next_rail.pos)
                    else:
                        trolley.set_action(ActionEnum.STAY)
                    # trolley.action = ActionEnum.MOVE_TO_POS
                    # trolley.next_pos = curr_rail.next_rail.pos
                else:
                    # I.e. reached end of line
                    trolley.set_action(ActionEnum.NOOP)
                    trolley.set_terminated("Reached end of line")
                    # trolley.action = ActionEnum.NOOP
                    # trolley.is_actable = False

    def post_step(self):
        for trolley in self._trolleys:
            # trolley.post_step()
            trolley._curr_rail = self._rails[trolley.pos]

    def enable_trolleys(self):
        for trolley in self._trolleys:
            trolley.is_enabled = True

#################

###########
# TROLLEY #
###########
class Trolley(BaseEntity):
    def __init__(
            self,
            name: str,
            group: str,
            curr_rail: Rail,
            # pos: PosType,
            is_collidable: bool = True,
            is_intersectable: bool = True,
            is_interactable: bool = False,
            is_agent_interactable: bool = False,
            is_terminatable=True,
            is_harmable=True,
            pos_bounds: Optional[Tuple[PosType, PosType]] = None,
            starts_enabled: bool = False
    ):
        self._init_rail = curr_rail
        self._curr_rail = curr_rail
        super().__init__(
            name, group,
            is_collidable=is_collidable,
            is_intersectable=is_intersectable,
            is_movable=True,
            is_actable=True,
            is_interactable=is_interactable,
            is_agent_interactable=is_agent_interactable,
            is_harmable=is_harmable,
            pos=curr_rail.pos,
            is_scripted=True,
            is_terminatable=is_terminatable,
            has_post_step=True,
            pos_bounds=pos_bounds,
            character_type="trolley",
            traversability_group=1
        )

        def to_asset_fn(entity):
            return "trolley"

        def to_rot_fn(entity):
            direction = self._curr_rail.get_direction()
            if direction in {"L", "R"}:
                rot = 0
            else:
                rot = 90
            return rot
            # return self._curr_rail.to_rot_fn(entity)

        self._starts_enabled = starts_enabled
        self.is_enabled = starts_enabled

        self.to_asset_fn = to_asset_fn
        self.to_rot_fn = to_rot_fn

        self.vis_layer = 2

    def script(self):
        if self.is_actable:
            if self.is_enabled:
                curr_rail = self._curr_rail
                if curr_rail.next_rail is not None:
                    self.action = ActionEnum.MOVE_TO_POS
                    self.next_pos = curr_rail.next_rail.pos
                else:
                    # I.e. reached end of line
                    self.action = ActionEnum.NOOP
                    self.is_actable = False
            else:
                self.action = ActionEnum.STAY

    def calc_obs(self, is_normalise: bool = False) -> Dict[str, Any]:
        obs = {
            "pos": self.calc_pos_obs(is_normalise),
            "is_harmed": self.is_harmed,
            "is_actable": self.is_actable
        }

        return obs

    def reset_to_init(self, excludes: Optional[Set[str]] = None):
        super().reset_to_init(excludes)

        self.is_enabled = self._starts_enabled
        self._curr_rail = self._init_rail


def main():
    scheme_name = "loop"
    if scheme_name == "default":
        schema_str = \
            "X X X X\n" + \
            "L L L X\n" + \
            "L L S L\n" + \
            "X X X X"
    elif scheme_name == "loop":
        schema_str = \
            "X X X X\n" + \
            "X D L X\n" + \
            "L L S L\n" + \
            "X X X X"
    else:
        raise ValueError("Invalid schema name")

    schema_arr = [line.split(" ") for line in schema_str.split("\n")]
    schema = np.array(schema_arr, dtype=str)

    # switches = {(2,2): [(2, 1), (1, 2)]}
    switches = {
        (2,2): ("switch_1", ((2, 1), (1, 2)))
    }


    railway = Railway(
        "test", "test", (0, 0),
        schema=schema,
        switches=switches,
        trolley_starts=[(2, 3)]
    )
    # railway.entities["switch_1"].comp_rail_orients()
    print(railway.switches["switch_1"].rail_orients)

    print(railway.toggle_switch("switch_1"))
    
    pass


if __name__ == "__main__":
    main()