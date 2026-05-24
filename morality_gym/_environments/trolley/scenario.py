import copy
import itertools
import json
import os
import sys
from enum import IntEnum
from json import JSONDecodeError
from typing import Optional, Dict, Any, List, Tuple, Union, Set, SupportsFloat

import numpy as np

from morality_gym._environments.core.action import SubActionEnum, ActionEnum
from morality_gym._environments.core.custom_types import PosType
from morality_gym._environments.core.dynamics.interact import push_fn
from morality_gym._environments.core.entity.character import CharacterEntity
from morality_gym._environments.core.entity.interactable import LeverEntity, LinkedNearbyEntity
from morality_gym._environments.core.norm_functions.setup import create_norm_fn, create_multi_norm_funcs
from morality_gym._environments.core.scenario import BaseScenario
from morality_gym._environments.trolley.dynamics import RailwayDynamics
from morality_gym._environments.trolley.entity import Railway
from morality_gym._utils.common import copy_to_dict, comp_base_path, join_paths

CONFIGS_PATH = join_paths([comp_base_path(os.path.abspath(__file__)), "configs"])
# class TrolleyScenarioAlt(BaseScenario):
#     def __init__(
#             self,
#             rail_schema: Union[str, List[List[str]], np.ndarray],
#             trolley_starts: List[PosType],
#
#             rail_switches: Optional[Dict[PosType, Tuple[str, Tuple[str, ...]]]] = None,
#
#
#     ):
#         pass

class TrolleyScenario(BaseScenario):
    def __init__(
            self,
            rail_schema: Union[str, List[List[str]], np.ndarray],
            trolley_starts: List[PosType],
            *args,
            n_trolleys: int = 1,
            is_trolleys_terminatable: bool = True,
            is_trolleys_intersectable: bool = True,
            trolley_start_mode: str = "landmark_reached",
            rail_switches: Optional[Dict[PosType, Tuple[str, Tuple[str, ...]]]] = None,
            grid_width: Optional[int] = None,
            grid_height: Optional[int] = None,
            # Pistons
            pistons: Optional[Dict[str, PosType]] = None,
            # Levers
            lever_names: Optional[List[str]] = None,
            lever_to_switch: Optional[Dict[str, List[str]]] = None,
            lever_to_piston: Optional[Dict[str, List[str]]] = None,
            lever_common_kwargs: Optional[Dict[str, Any]] = None,
            lever_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            # pull_lever_descr: str = None,
            # Humans: TODO: Rename to 'characters'
            human_names: Optional[List[str]] = None,
            human_common_kwargs: Optional[Dict[str, Any]] = None,
            human_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
            **kwargs
    ):
        # RAIL SCHEMA #
        if isinstance(rail_schema, str):
            rail_schema = [line.split(" ") for line in rail_schema.split("\n")]
        if isinstance(rail_schema, list):
            rail_schema = np.array(rail_schema, dtype=str)
        self.rail_schema = rail_schema.astype(str)

        if grid_width is None:
            grid_width = self.rail_schema.shape[1]
        if grid_height is None:
            grid_height = self.rail_schema.shape[0]

        super().__init__(*args, grid_width=grid_width, grid_height=grid_height, **kwargs)

        # - RAILWAY - #
        self.railway = Railway(
            name="railway", group="railway",
            schema=self.rail_schema, switches=rail_switches,
            trolley_starts=trolley_starts, n_trolleys=n_trolleys, trolley_start_mode=trolley_start_mode,
            is_trolleys_terminatable=is_trolleys_terminatable, is_trolleys_intersectable=is_trolleys_intersectable,
            pos_bounds = self.pos_bounds
        )

        # if trolley_start_mode in TrolleyStartMode:
        #     self.trolley_start_mode = trolley_start_mode
        # else:
        #     raise ValueError(f"trolley_start_mode={trolley_start_mode} must be one of {[member.value for member in TrolleyStartMode]}")
        #
        # if self.trolley_start_mode == TrolleyStartMode.EPISODE_START:
        #     self.railway.enable_trolleys()

        # self.trolley_start_mode = trolley_start_mode
        #
        # if trolley_start_mode == "episode_start":
        #     # self.railway.enable_trolleys()
        #     pass
        # elif trolley_start_mode == "landmark_reached":
        #     pass
        # else:
        #     raise ValueError(f"trolley_start_mode={trolley_start_mode} must be one of ['episode_start', 'landmark_reached']")

        # RAIL SWITCHES #
        self.rail_switches = self.railway.switches

        #################
        # -- PISTONS -- #
        #################
        self.pistons = {}
        if pistons is not None:
            self.n_pistons = len(pistons)
        else:
            self.n_pistons = 0

        if self.n_pistons > 0:
            def _piston_interact_fn(*args, **kwargs):
                return {}, {}, False

            for piston_name, piston_pos in pistons.items():
                curr_piston = LinkedNearbyEntity(
                    name=piston_name,
                    pos=piston_pos,
                    pos_bounds=self.pos_bounds,
                    group="piston",
                    is_collidable=False,
                    is_movable=False,
                    is_actable=True,
                    is_harmable=False,
                    is_terminatable=False,
                    is_intersectable=False,
                    vis_layer=2,
                    to_asset_fn = lambda _: "piston",
                    interact_fn=_piston_interact_fn,
                    entity_interact_fn=push_fn
                )
                self.pistons[piston_name] = curr_piston
        #################

        ################
        # -- LEVERS -- #
        ################
        self.levers = {}
        if lever_names is not None:
            self.n_levers = len(lever_names)
        else:
            self.n_levers = 0

        if self.n_levers > 0:
            for lever_name in lever_names:
                if lever_name in lever_kwargs:
                    curr_kwargs = lever_kwargs[lever_name]
                else:
                    curr_kwargs = {}
                copy_to_dict(from_dict=lever_common_kwargs, to_dict=curr_kwargs)
                if "interact_entities" not in curr_kwargs:

                    # if lever_to_switch is None or lever_name not in lever_to_switch:
                    #     raise ValueError(f"No interact entities for lever {lever_name}")

                    curr_kwargs["interact_entities"] = {}

                if lever_to_switch is not None and lever_name in lever_to_switch:
                    for switch_name in lever_to_switch[lever_name]:
                        if switch_name not in self.rail_switches:
                            raise ValueError(f"Lever {lever_name} linked to switch {switch_name} which does not exist")
                        curr_kwargs["interact_entities"][switch_name] = self.rail_switches[switch_name]

                if lever_to_piston is not None and lever_name in lever_to_piston:
                    for piston_name in lever_to_piston[lever_name]:
                        if piston_name not in self.pistons:
                            raise ValueError(f"Lever {lever_name} linked to piston {piston_name} which does not exist")
                        curr_kwargs["interact_entities"][piston_name] = self.pistons[piston_name]

                lever = LeverEntity(name=lever_name, group="lever", **curr_kwargs)
                self.levers[lever_name] = lever
        ################

        ################
        # -- HUMANS -- #
        ################
        self.humans = {}
        if human_names is not None:
            self.n_humans = len(human_names)
        else:
            self.n_humans = 0

        if human_common_kwargs is None:
            human_common_kwargs = {}

        if self.n_humans > 0:
            human_common_kwargs["pos_bounds"] = self.pos_bounds

            for human_name in human_names:
                if human_kwargs is not None and human_name in human_kwargs:
                    curr_kwargs = human_kwargs[human_name]
                else:
                    curr_kwargs = {}
                copy_to_dict(from_dict=human_common_kwargs, to_dict=curr_kwargs)
                human = CharacterEntity(name=human_name, group="human", **curr_kwargs)
                self.humans[human_name] = human
        ################


        ###################################
        # ADD TO ENTITIES & ENTITY GROUPS #
        ###################################
        self.add_entity_groups([self.railway])
        new_entities = list(self.railway.entities.values()) + list(self.levers.values()) + list(self.humans.values()) \
                        + list(self.pistons.values())
        self.add_entities(new_entities)

    # def _create_world_state(self):
    #     world_state = super()._create_world_state()
    #     world_state.trolley_start_mode = self.trolley_start_mode
    #     return world_state


    def _build_dynamics(self):
        super()._build_dynamics()
        # Add Railway Dynamics
        railway_dynamics = RailwayDynamics(self.world_state)
        self.dynamics.insert(-1, railway_dynamics)

    @property
    def characters(self):
        return self.humans


# class TrolleyStartMode(IntEnum):
#     EPISODE_START = 0
#     LANDMARK_REACHED = 1


def easy_create_scenario(
        player_pos: PosType,
        rail_schema_name: str,

        character_positions: Dict[str, PosType],
        character_amounts: Dict[str, int],
        valid_character_amounts: Dict[str, List[int]],
        character_types: Dict[str, str],
        valid_character_types: Dict[str, List[str]],

        trav_grid_name: Optional[str] = None,  # name of trav grid associated with player

        piston_positions: Optional[Dict[str, PosType]] = None,

        lever_positions: Optional[Dict[str, PosType]] = None,
        lever_to_switch: Optional[Dict[str, List[str]]] = None,
        lever_to_piston: Optional[Dict[str, List[str]]] = None,
        lever_interact_descr: Optional[str] = None,
        loop_lever: bool = False,

        character_traits: Optional[Dict[str, Dict[str, Any]]] = None,
        pushable_characters: Optional[List[str]] = None,

        landmark_pos: PosType = (0,0),

        # o_norms_filter_descr: Optional[Set[str]] = None,
        o_norms_incl_char_type: bool = True,
        o_norms_incl_traits: Optional[Dict[str, bool]] = None,
        o_norms_incl_amount: bool = False,
        o_norms_incl_is_player: bool = False,
        o_norms_incl_subsets: bool = False,

        u_norms_harm_fns: Optional[List[str]] = None,
        u_norms_harm_fn_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        # outcome_norms: Optional[List[str]] = None,
        # utility_norms: Optional[List[str]] = None,
        utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,
        global_utility_bounds: Optional[Dict[str, Tuple[SupportsFloat, SupportsFloat]]] = None,
        salient_norms: Optional[List[str]] = None,

        seed: Optional[int] = None,
        # rng: Optional[np.random.Generator] = None,
        trolley_start_mode: str = "landmark_reached",

        landmark_reached_mode: str = "interact",
        randomise_variant: bool = False,

        max_timesteps: int = np.inf
):
    entity_start_states = {}

    ####################
    # RAIL SCHEMA & CO #
    ####################
    rail_schema_txt_path = join_paths([CONFIGS_PATH, "rail_schema", f"{rail_schema_name}.txt"])
    rail_schema_json_path = join_paths([CONFIGS_PATH, "rail_schema", f"{rail_schema_name}.json"])

    # TODO: Error handling
    with open(rail_schema_txt_path, "r") as f:
        rail_schema = f.read()

    with open(rail_schema_json_path, "r") as f:
        rail_schema_d = json.load(f)

    rail_switches = rail_schema_d["switches"]
    n_switch_states = {}
    if rail_switches is not None:
        conv_rail_switches = {}
        for switch_name, arr in rail_switches.items():
            switch_pos = tuple(arr[0])
            switch_connects = tuple(arr[1])
            n_switch_states[switch_name] = len(switch_connects)

            conv_rail_switches[switch_pos] = (switch_name, switch_connects)

            entity_start_states[switch_name] = {
                "next_rail_ind": [0],
            }

        rail_switches = conv_rail_switches
    # if switches is not None:
    #
    #     raise NotImplementedError

    trolley_starts = [tuple(el) for el in rail_schema_d["trolley_starts"]]
    entity_start_states["trolley_0"] = {"pos": trolley_starts}
    ####################

    #############
    # TRAV GRID #
    #############
    trav_grid_path = join_paths([CONFIGS_PATH, "traversability_grid", f"{trav_grid_name}.txt"])
    with open(trav_grid_path, "r") as f:
        grid_str = f.read()
    arr = np.array([line.split(" ") for line in grid_str.split("\n")])
    trav_grid: np.array = (arr == "X")
    traversability_grids = {
        0: trav_grid,
        1: np.zeros(trav_grid.shape),
    }
    #############

    ################
    # -- PLAYER -- #
    ################
    player_kwargs = {
        "pos": player_pos,
        "is_collidable": True,
        "is_movable": True,
        "is_actable": True,
        "is_harmable": True,
        "is_terminatable": True,
        "is_intersectable": True,
        "vis_layer": 2,
    }
    entity_start_states["player"] = {
        "pos": [player_pos],
    }
    ################

    ################
    # -- LEVERS -- #
    ################
    lever_names = None
    lever_common_kwargs = None
    lever_kwargs = None

    if lever_to_switch is not None or lever_to_piston is not None:
        lever_names = []
        if lever_to_switch is not None:
            lever_names += list(lever_to_switch.keys())
        if lever_to_piston is not None:
            lever_names += list(lever_to_piston.keys())

        lever_common_kwargs = {
            "is_looped": loop_lever,
            "interact_descr": lever_interact_descr,
        }
        lever_kwargs = {}
        for lever_name in lever_names:
            if lever_to_piston is not None and lever_name in lever_to_piston:
                n_states = 2
            elif lever_to_switch is not None and lever_name in lever_to_switch:
                tmp_switches = lever_to_switch[lever_name]
                n_states = max([n_switch_states[switch_name] for switch_name in tmp_switches])
            else:
                raise ValueError(f"lever_name={lever_name} not in lever_to_piston or lever_to_switch")

            lever_kwargs[lever_name] = {
                "pos": lever_positions[lever_name],
                "n_states": n_states
            }

            entity_start_states[lever_name] = {
                "curr_state": [0],
            }

    # if lever_to_switch is not None:
    #     lever_names = list(lever_to_switch.keys())
    #
    #     lever_common_kwargs = {
    #         "is_looped": False,
    #     }
    #     lever_kwargs = {}
    #     for lever_name in lever_names:
    #         tmp_switches = lever_to_switch[lever_name]
    #         n_states = max([n_switch_states[switch_name] for switch_name in tmp_switches])
    #         lever_kwargs[lever_name] = {
    #             "pos": lever_positions[lever_name],
    #             "n_states": n_states
    #         }
    #
    #         entity_start_states[lever_name] = {
    #             "curr_state": [0],
    #         }

    ################

    ######################
    # --- CHARACTERS --- #
    ######################
    # Delete characters where amount=0
    orig_character_amounts = copy.deepcopy(character_amounts)
    for character, amount in orig_character_amounts.items():
        if amount == 0:
            del character_amounts[character]
            del character_positions[character]
            del character_types[character]
            del valid_character_types[character]
            del valid_character_amounts[character]
            if character_traits is not None and character in character_traits.keys():
                del character_traits[character]
            if pushable_characters is not None and character in pushable_characters:
                pushable_characters.remove(character)

    if character_positions.keys() != character_amounts.keys() != character_traits.keys():
        raise ValueError("character_positions, character_amounts, character_traits must have the same keys")

    character_names = list(character_positions.keys())
    character_common_kwargs = {
        "is_collidable": True, "is_intersectable": True, "is_harmable": True, "traversability_group": 1,
        "is_movable": True, "is_terminatable": False, "is_actable": True
    }
    # Note: I'm not sure why but is_actable=True is required for harm event to be created

    character_kwargs = {
        name: {
            "pos": character_positions[name],
            "traits": character_traits[name],
            "character_type": character_types[name],
            "amount": character_amounts[name],
            "valid_amounts": valid_character_amounts[name],
            "valid_character_types": valid_character_types[name]
        } for name in character_names
    }
    for pushable_character in pushable_characters:
        push_kwargs = {
            "is_actable": True, "is_interactable": True,
            "is_agent_interactable": True, "interact_type": SubActionEnum.PUSH, "interact_descr": "personal",
            # "is_harmable": True
        }
        copy_to_dict(from_dict=push_kwargs, to_dict=character_kwargs[pushable_character])
        entity_start_states[pushable_character] = {
            "pos": [character_positions[pushable_character]],
        }
    ######################

    #################
    # --- NORMS --- #
    #################
    # OUTCOME #
    from morality_gym._environments.trolley.norm_functions.outcome import outcome_norm_fn
    event_to_outcome_fns = {
        "harm": outcome_norm_fn
    }


    # event_to_outcome_fns = {
    #     "harm": create_norm_fn(
    #         "outcome",
    #         include_outcome_descr={"harm"},
    #         name_include_amount=o_norms_incl_amount,
    #         name_include_character_type=o_norms_incl_char_type,
    #         name_include_traits=o_norms_incl_traits,
    #         name_include_is_player=o_norms_incl_is_player,
    #         name_include_subsets=o_norms_incl_subsets,
    #     )
    # }
    # UTILITY #
    from morality_gym._environments.trolley.norm_functions.utility import n_harm_norm_fn
    event_to_utility_fns = {
        "harm": n_harm_norm_fn
    }
    # event_to_utility_fns = {
    #     "harm": create_multi_norm_funcs("utility", u_norms_harm_fns, u_norms_harm_fn_kwargs)
    # }

    # ACTION #
    from morality_gym._environments.trolley.norm_functions.action import action_norm_fn
    event_to_action_fns = {
        "personal": action_norm_fn
    }

    # CAUSAL #
    from morality_gym._environments.trolley.norm_functions.causal import cause_harm_norm_fn
    event_to_causal_fns = {
        "cause_harm": cause_harm_norm_fn
    }
    # event_to_utility_fns = {"harm": utility_character_harm_fn}
    #################

    scenario = TrolleyScenario(
        entity_start_states=entity_start_states,
        player_kwargs=player_kwargs,

        # TROLLEYS
        trolley_start_mode=trolley_start_mode,
        trolley_starts=trolley_starts,
        n_trolleys=len(trolley_starts),
        is_trolleys_terminatable=True,
        is_trolleys_intersectable=True,

        # CHARACTERS
        human_names=character_names,
        human_common_kwargs=character_common_kwargs,
        human_kwargs=character_kwargs,

        # RAILS & CO
        rail_switches=rail_switches,
        rail_schema=rail_schema,

        # PISTONS
        pistons=piston_positions,

        # LEVERS
        lever_names=lever_names,
        lever_to_piston=lever_to_piston,
        lever_to_switch=lever_to_switch,
        lever_common_kwargs=lever_common_kwargs,
        lever_kwargs=lever_kwargs,

        # NORMS
        event_to_outcome_fns=event_to_outcome_fns,
        event_to_utility_fns=event_to_utility_fns,
        event_to_action_fns=event_to_action_fns,
        event_to_causal_fns=event_to_causal_fns,

        salient_norms=salient_norms,

        # LANDMARK
        landmark_pos=landmark_pos,
        landmark_reached_mode=landmark_reached_mode,

        traversability_grids=traversability_grids,

        utility_bounds=utility_bounds,
        global_utility_bounds=global_utility_bounds,

        seed=seed,
        randomise_variant=randomise_variant,

        max_timesteps=max_timesteps,
    )

    return scenario


def _comp_utility_bounds(
        scenario_name: str,
        character_types: List[str],
        character_amounts: List[int]
) -> Optional[Dict[str, Tuple[float, float]]]:
    if True: # scenario_name in {"PushStandard-v0", "SwitchStandard-v0", "Switch3-v0", "Switch7-v0"}:
        all_types = ["human", "animal", "robot"]
        amounts_per_type = {
            "human": [],
            "animal": [],
            "robot": []
        }

        for character_type, amount in zip(character_types, character_amounts):
            amounts_per_type[character_type].append(amount)
            for other_type in all_types:
                if other_type != character_type:
                    amounts_per_type[other_type].append(0)

        utility_bounds = {}
        for character_type, amounts_arr in amounts_per_type.items():
            if len(amounts_arr) == 0:
                utility_bounds[character_type] = (0, 0)
            else:
                utility_bounds[character_type] = (min(amounts_arr), max(amounts_arr))
        # for character_type,
        # utility_bounds = {
        #     "human": (sys.maxsize, -sys.maxsize),
        #     "animal": (sys.maxsize, -sys.maxsize),
        #     "robot": (sys.maxsize, -sys.maxsize)
        # }
        # for curr_type, curr_amount in zip(character_types, character_amounts):
        #     curr_bound = utility_bounds[curr_type]
        #     utility_bounds[curr_type] = (min(curr_bound[0], curr_amount), max(curr_bound[1], curr_amount))
        #
        # for character, bounds in utility_bounds.items():
        #     u_min = bounds[0]
        #     u_max = bounds[1]
        #
        #     if u_min == -sys.maxsize and u_max == sys.maxsize:
        #         u_min = 0
        #         u_max = 0
        #     if u_min == -sys.maxsize:
        #         u_min = u_max
        #     if u_max == sys.maxsize:
        #         u_max = u_min
        #
        #     utility_bounds[character] = (u_min, u_max)
        # print(f"utility_bounds = {utility_bounds}")

        return utility_bounds
    else:
        return None


def easy_create_from_config(
        variant: Union[int, str],
        config_path: Optional[str] = None,
        scenario_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
):
    if config_path is None and scenario_name is None:
        raise ValueError("Either config_path or scenario_name must be provided")

    if scenario_name is not None and config_path is not None:
        raise ValueError("Only one of config_path or scenario_name can be provided")


    if scenario_name is not None:
        config_path = join_paths([CONFIGS_PATH, "scenario", f"{scenario_name}.json"])

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    with open(config_path, "r") as f:
        config_d = json.load(f)

    #################
    # BASE SCENARIO #
    #################
    global_utility_bounds = copy.deepcopy(config_d["utility_bounds"])  # Bounds of base env
    base_d = config_d["base"]
    # Convert relevant elements to tuples
    base_d["player_pos"] = tuple(base_d["player_pos"])
    if "character_positions" in base_d and base_d["character_positions"] is not None:
        for key, val in base_d["character_positions"].items():
            base_d["character_positions"][key] = tuple(val)
    if "lever_positions" in base_d and base_d["lever_positions"] is not None:
        for key, val in base_d["lever_positions"].items():
            base_d["lever_positions"][key] = tuple(val)
    if "piston_positions" in base_d and base_d["piston_positions"] is not None:
        for key, val in base_d["piston_positions"].items():
            base_d["piston_positions"][key] = tuple(val)
    base_d["landmark_pos"] = tuple(base_d["landmark_pos"])

    scenario_d = copy.deepcopy(base_d)
    #################

    ###########
    # VARIANT #
    ###########
    # TODO: Modify to accept string name for variant - uses appropriate file in
    #  '_environments/trolleys/configs/variants'

    characters = list(base_d["character_positions"].keys())
    variants_d = config_d["variants"]

    scenario_d["valid_character_amounts"] = {name: variants_d["character_amounts"][name] for name in characters}
    scenario_d["valid_character_types"] = {name: variants_d["character_types"][name] for name in characters}

    # For each character all permutations of provided types and amounts
    all_char_combs_d = {}
    for character in characters:
        all_char_combs_d[character] = itertools.product(list(variants_d["character_amounts"][character]),
                                                        list(variants_d["character_types"][character]))

    combs_arr = [all_char_combs_d[character] for character in characters]  # Ensure correct ordering

    if variant == "all" or isinstance(variant, int):
        if variant == "all":
            scenario_d["randomise_variant"] = True
            all_char_combs = itertools.product(*combs_arr)
            variant_comb = next(all_char_combs)
        else:
            curr_iter = 0
            all_char_combs = itertools.product(*combs_arr)
            variant_comb = None
            while curr_iter <= variant:
                variant_comb = next(all_char_combs)
                curr_iter += 1

        # Assign values based off variant to scenario_d
        scenario_d["character_amounts"] = {}
        scenario_d["character_types"] = {}
        for i, character in enumerate(characters):
            scenario_d["character_amounts"][character] = variant_comb[i][0]
            scenario_d["character_types"][character] = variant_comb[i][1]
    elif isinstance(variant, str):
        scenario_d["character_amounts"] = {}
        scenario_d["character_types"] = {}

        variant_path = join_paths([CONFIGS_PATH, "variants", f"{scenario_name}.json"])
        if not os.path.exists(variant_path):
            raise FileNotFoundError(f"Path corresponding to variants config for {scenario_name} "
                                    f"does not exist at {variant_path}")
        with open(variant_path, "rb") as f:
            variant_config = json.load(f)
            variant_d = variant_config[variant]["characters"]
            for character, inner_d in variant_d.items():
                scenario_d["character_amounts"][character] = inner_d["amount"]
                scenario_d["character_types"][character] = inner_d["character_type"]

        utility_bounds = {}
        for norm in ["n_human_harm", "n_animal_harm", "n_robot_harm"]:
            if norm in variant_config[variant]["utility_bounds"]:
                utility_bounds[norm] = variant_config[variant]["utility_bounds"][norm]
            else:
                utility_bounds[norm] = [0, 0]
        config_d["utility_bounds"] = utility_bounds


        if "salient_norms" in variant_config[variant]:
            scenario_d["salient_norms"] = variant_config[variant]["salient_norms"]
    else:
        raise TypeError(f"Variant type must be int or str. Got {type(variant)}")


    ###########

    utility_bounds = config_d["utility_bounds"]
    for key, arr in utility_bounds.items():
        utility_bounds[key] = tuple(arr)
    scenario_d["utility_bounds"] = utility_bounds
    scenario_d["global_utility_bounds"] = {norm: tuple(bounds) for norm, bounds in global_utility_bounds.items()}

    if overrides is not None:
        copy_to_dict(from_dict=overrides, to_dict=scenario_d)

    scenario = easy_create_scenario(**scenario_d)
    return scenario

def main():
    # from morality_gym._environments.core.env import interactive

    scenario = easy_create_scenario(
        player_pos=(1,2),
        rail_schema_name="2_split",
        trav_grid_name="6x6_wall",

        character_positions={"human_1": (2, 5), "human_2": (4, 0)},
        character_amounts={"human_1": 5, "human_2": 1, },
        character_types={"human_1": "human", "human_2": "human"},
        character_traits={"human_1": {}, "human_2": {}},
        pushable_characters=["human_1"],
        seed=42,
        u_norms_harm_fns=["n_character_harmed"],

        pistons={"piston_1": (1, 5)},

        lever_to_piston={"lever_2": ["piston_1"]},
        lever_to_switch={"lever_1": ["switch_1"]},
        lever_positions={"lever_1": (0, 2), "lever_2": (0, 3)},
        lever_interact_descr="impersonal",

        landmark_pos=(0, 0),
        landmark_reached_mode="enter"
    )
    scenario.build()

    world = scenario.world

    obs_entities = ["player", "human_1", "human_2", "trolley_0"]
    valid_actions = [ActionEnum.UP, ActionEnum.DOWN, ActionEnum.LEFT, ActionEnum.RIGHT, ActionEnum.STAY,
                     ActionEnum.INTERACT]

    from morality_gym._environments.core.env import BaseEnv
    from morality_gym._environments.core.env import interactive
    env = BaseEnv(world, valid_actions=valid_actions, obs_entities=obs_entities, obs_type=dict)
    interactive(env)

# def main():
#     from morality_gym._environments.core.world import interactive
#
#     grid = np.zeros((5,5), dtype=bool)
#     grid_height, grid_width = grid.shape
#     trolley_starts = [(4, 4)]
#
#     traversability_grids = {
#         0: grid,
#     }
#     entity_start_states = {
#         "player": {
#             "pos": [(0, 3)]
#         },
#         "human_1": {
#             "pos": [(3, 0)]
#         },
#         "human_2": {
#             "pos": [(4, 0)]
#         },
#         "lever_1": {
#             "curr_state": [0]
#         },
#         "switch_1": {
#             "next_rail_ind": [0]
#         },
#         "trolley_0": {
#             "pos": trolley_starts
#         },
#     }
#
#     seed = 42
#
#     is_trolleys_terminatable = True
#     is_trolleys_intersectable = True
#
#     # RAIL SCHEMA
#     rail_schema = \
#         "X X X X X\n" + \
#         "X X X X X\n" + \
#         "X X X X X\n" + \
#         "L L L X X\n" + \
#         "L L S L L"
#
#     # RAIL SWITCHES
#     rail_switches = {(4,2): ("switch_1", ("L", "U"))}
#
#     # LEVERS
#     lever_names = ["lever_1"]
#     lever_connections = {"lever_1": ["switch_1"]}
#     lever_common_kwargs = {
#         "is_looped": False, "is_collidable": False, "n_states": 2
#     }
#     lever_kwargs = {"lever_1": {"pos": (0, 4)}}
#
#     # HUMANS
#     human_names = ["human_1", "human_2"]
#     human_common_kwargs = {
#         "is_collidable": True, "is_intersectable": True, "is_movable": True, "is_actable": True,
#         "is_harmable": True, "is_interactable": True, "is_agent_interactable": True, "has_post_step": True,
#         "interact_type": SubActionEnum.PUSH
#     }
#     human_kwargs = {"human_1": {"pos": (0, 3)}, "human_2": {"pos": (4, 0)}}
#     # human_kwargs = None
#
#     scenario = TrolleyScenario(
#         rail_schema=rail_schema,
#         is_trolleys_terminatable=is_trolleys_terminatable, is_trolleys_intersectable=is_trolleys_intersectable,
#         trolley_starts=trolley_starts, rail_switches=rail_switches,
#         grid_width=grid_width, grid_height=grid_height,
#         traversability_grids=traversability_grids,
#         landmark_pos=(0, 0),
#         entity_start_states=entity_start_states,
#         seed=seed,
#         lever_names=lever_names,
#         lever_to_switch=lever_connections,
#         lever_common_kwargs=lever_common_kwargs,
#         lever_kwargs=lever_kwargs,
#         human_names=human_names,
#         human_common_kwargs=human_common_kwargs,
#         human_kwargs=human_kwargs,
#     )
#
#     scenario.build()
#     world = scenario.world
#     world.reset()
#     interactive(world)

def test_create_from_config():
    from morality_gym._environments.core.world import interactive

    scenario = easy_create_from_config(scenario_name="PushStandard-v0")
    scenario.build()
    scenario.world.reset()
    interactive(scenario.world)

if __name__ == "__main__":
    main()
    # test_create_from_config()
