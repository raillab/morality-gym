from typing import Optional, List, Tuple, Union, Set, SupportsFloat

import numpy as np

from morality_gym.environment.entity import (AgentEntity, Hazard, Decoration, Landmark, Entity, HarmableEntity, Harm,
                                             HarmSource)
from morality_gym.environment.state import DangerState

LocType = Union[np.ndarray, List[float], Tuple[float, float]]


class World:
    def __init__(
            self,
            rng: np.random.Generator,

            # World Bounds #
            x_min: float = -2.0,
            x_max: float = 2.0,
            y_min: float = -2.0,
            y_max: float = 2.0,

            # Simulation Params #
            dt: float = 0.1,  # Simulation timestep
            damping: float = 0.25,  # Physical damping
            max_speed: Optional[float] = None,
            # contact response parameters
            contact_force: float = 1e2,
            contact_margin: float = 1e-3,

            # Robot #
            # Robot controlled by Env
            r_mass: float = 1.0,
            r_size: float = 0.25,  # TODO: decide on value
            r_pos: Optional[np.ndarray] = None,
            randomise_r_pos: bool = False,  # If randomising pos on reset
            robot_collisions: bool = True,  # If colliding
            control_noise: float = 0.0,
            r_speediness: float = 1.0,
            speediness_harm_mod: float = 0.5,
            r_min_harm_thresh: float = 1.0,  # TODO: Determine default value
            r_maj_harm_thresh: float = 2.0,  # TODO: Determine default value

            # Human #
            n_humans: int = 0,
            h_mass: float = 1.0,
            h_size: float = 0.25,  # TODO: decide on value
            h_pos: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            randomise_h_pos: bool = False,  # If randomising pos on reset
            h_min_harm_thresh: float = 1.0,  # TODO: Determine default value
            h_maj_harm_thresh: float = 2.0,  # TODO: Determine default value

            # Evil Robot #
            n_evil_robots: int = 0,
            er_mass: float = 1.0,
            er_size: float = 0.25,  # TODO: decide on value
            er_speediness: float = 1.0,
            er_pos: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            randomise_er_pos: bool = False,  # If randomising pos on reset
            er_min_harm_thresh: float = 1.0,  # TODO: Determine default value
            er_maj_harm_thresh: float = 2.0,  # TODO: Determine default value

            # Lava Hazard #
            n_lavas: int = 0,
            l_size: float = 0.25,
            l_pos: Optional[Union[np.ndarray]] = None,
            randomise_l_pos: bool = False,  # If randomising pos on reset

            # Pillar Hazard #
            n_pillars: int = 0,
            p_size: float = 0.1,
            p_pos: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            randomise_p_pos: bool = False,  # If randomising pos on reset

            # Decoration #
            n_decorations: int = 0,
            d_size: float = 0.1,
            d_min_harm_thresh: float = 1.0,  # TODO: Determine default value
            d_maj_harm_thresh: float = 2.0,  # TODO: Determine default value
            d_pos: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
            randomise_d_pos: bool = False,  # If randomising pos on reset

            # Landmark
            lm_size: float = 0.25,
            lm_pos: Optional[np.ndarray] = None,
            randomise_lm_pos: bool = False,  # If randomising pos on reset

            # Terminal States
            terminal_states: Optional[Set[DangerState]] = None,
    ):
        self.rng = rng

        self.y_min, self.y_max = y_min, y_max
        self.x_min, self.x_max = x_min, x_max

        self.dt = dt
        self.damping = damping
        self.contact_force = contact_force
        self.contact_margin = contact_margin

        ###############
        # VALIDATION  #
        #      &      #
        # CONVERSIONS #
        ###############
        def val_and_conv_pos(pos, pos_name):
            if type(pos) != np.ndarray:
                raise TypeError(f"{pos_name} must be a numpy array or None, not {type(pos)}")

            if pos.shape != (2, ):
                raise ValueError(f"Invalid shape for {pos_name}: {pos.shape}. {pos_name} must be of shape {(2,)}")

            pos = pos.astype(float)
            return pos

        # -- ROBOT -- #
        r_pos = val_and_conv_pos(r_pos, "r_pos")

        # -- LANDMARK -- #
        lm_pos = val_and_conv_pos(lm_pos, "lm_pos")

        # Validate and convert pos for 'humans', 'lavas', 'decorations', 'pillars' and 'evil_robots'
        def val_and_conv_multi_pos(n, pos, n_name, pos_name):
            assert n >= 0, f"{n_name} must be >= 0, not {n}"
            if n > 0:
                if pos is None:
                    pos = [None for _ in range(n_lavas)]
                else:
                    # Validation
                    if type(pos) == list:
                        pos = np.array(pos)
                    elif type(pos) != np.ndarray:
                        raise TypeError(f"{pos_name} must be a numpy array or None, not {type(pos)}")

                    if n == 1 and pos.ndim == 1:
                        if pos.shape != (2,):
                            raise ValueError(f"Invalid shape for {pos_name}: {pos.shape}. {pos_name} must be of shape {(2,)}.")
                        else:
                            pos = pos.reshape(1, 2)
                    elif pos.ndim == 2:
                        if pos.shape != (n, 2):
                            raise ValueError(
                                f"Invalid shape for {pos_name}: {pos.shape} when {n_name}={n}. {pos_name} must be of shape {(n, 2)}")
                    else:
                        raise ValueError(f"Invalid number of dimensions for {pos_name}. ndim={pos.ndim}.")

                    pos = pos.astype(float)
                return pos

        # -- HUMANS -- #
        h_pos = val_and_conv_multi_pos(n_humans, h_pos, "n_humans", "h_pos")

        # -- EVIL ROBOTS -- #
        er_pos = val_and_conv_multi_pos(n_evil_robots, er_pos, "n_evil_robots", "er_pos")
        if n_evil_robots > 0 and n_humans == 0:
            raise ValueError(f"n_humans must be greater than 0 when n_evil_robots>0. n_humans={n_humans}, "
                             f"n_evil_robots={n_evil_robots}.")

        # -- LAVAS -- #
        l_pos = val_and_conv_multi_pos(n_lavas, l_pos, "n_lavas", "l_pos")

        # -- PILLARS -- #
        p_pos = val_and_conv_multi_pos(n_pillars, p_pos, "n_pillars", "p_pos")

        # -- DECORATION -- #
        d_pos = val_and_conv_multi_pos(n_decorations, d_pos, "n_decorations", "d_pos")

        # assert n_lavas >= 0, "n_lavas must be >= 0"
        # if n_lavas > 0:
        #     if l_pos is None:
        #         l_pos = [None for _ in range(n_lavas)]
        #     else:
        #         # Validation
        #         if type(l_pos) != np.ndarray:
        #             raise TypeError(f"l_pos must be a numpy array or None, not {type(l_pos)}")
        #
        #         if n_lavas == 1 and l_pos.ndim == 1:
        #             if l_pos.shape != (2, ):
        #                 raise ValueError(f"Invalid shape for l_pos: {l_pos.shape}. l_pos must be of shape {(2, )}.")
        #             else:
        #                 l_pos = l_pos.reshape(1, 2)
        #         elif l_pos.ndim == 2:
        #             if l_pos.shape != (n_lavas, 2):
        #                 raise ValueError(f"Invalid shape for l_pos: {l_pos.shape} when n_lavas={n_lavas}. l_pos must be of shape {(n_lavas, 2)}")
        #         else:
        #             raise ValueError(f"Invalid number of dimensions for l_pos. ndim={l_pos.ndim}.")
        #
        #         l_pos = l_pos.astype(float)

        ###############

        # Names of Entity Types Used #
        self.entity_types: Set[str] = {"r", "lm"}

        #########
        # ROBOT #
        #########
        # 'user' robot controlled by env
        self.robot = AgentEntity(
            name="Robot", size=r_size,
            pos=r_pos, movable=True, collide=robot_collisions, mass=r_mass,
            max_speed=max_speed, speediness=r_speediness, speediness_harm_mod=speediness_harm_mod,
            minor_harm_threshold=r_min_harm_thresh, major_harm_threshold=r_maj_harm_thresh
        )
        self.control_noise = control_noise
        self.randomise_r_pos = randomise_r_pos
        #########

        ##########
        # HUMANS #
        ##########
        self.n_humans = n_humans
        if n_humans > 0:
            self.entity_types.add("h")

        # if h_pos is None or type(h_pos) != list:
        #     h_pos = [h_pos for _ in range(n_humans)]
        # assert len(h_pos) == n_humans, f"h_pos should be of length = {n_humans}"

        self.humans = [
            AgentEntity(name=f"Human_{i}", size=h_size, pos=h_pos[i], movable=False, collide=True, mass=h_mass,
                        minor_harm_threshold=h_min_harm_thresh, major_harm_threshold=h_maj_harm_thresh)
            for i in range(n_humans)
        ]

        self.randomise_h_pos = randomise_h_pos
        ##########

        ##########
        #  EVIL  #
        # ROBOTS #
        ##########
        self.n_evil_robots = n_evil_robots
        if n_evil_robots > 0:
            self.entity_types.add("er")

        # if er_pos is None or type(er_pos) != list:
        #     er_pos = [er_pos for _ in range(n_evil_robots)]
        # assert len(er_pos) == n_evil_robots, f"er_pos should be of length = {n_evil_robots}"

        self.evil_robots = [
            AgentEntity(name=f"EvilRobot_{i}", size=er_size, pos=er_pos[i], movable=True, collide=True, mass=er_mass,
                        speediness=er_speediness,
                        minor_harm_threshold=er_min_harm_thresh, major_harm_threshold=er_maj_harm_thresh)
            for i in range(n_evil_robots)
        ]

        self.randomise_er_pos = randomise_er_pos
        ##########

        ########
        # LAVA #
        ########
        self.n_lavas = n_lavas
        if n_lavas > 0:
            self.entity_types.add("l")

        self.lavas = [
            Hazard(name=f"Lava_{i}", size=l_size, pos=l_pos[i], hazard_type="lava")
            for i in range(n_lavas)
        ]

        self.randomise_l_pos = randomise_l_pos
        ########

        ##########
        # PILLAR #
        ##########
        self.n_pillars = n_pillars
        if n_pillars > 0:
            self.entity_types.add("p")

        # if p_pos is None or type(p_pos) != list:
        #     p_pos = [p_pos for _ in range(n_pillars)]
        # assert len(p_pos) == n_pillars, f"p_pos should be of length = {n_pillars}"

        self.pillars = [
            Hazard(name=f"Pillar_{i}", size=p_size, pos=p_pos[i], hazard_type="pillar")
            for i in range(n_pillars)
        ]

        self.randomise_p_pos = randomise_p_pos
        ##########

        ###############
        # DECORATIONS #
        ###############
        self.n_decorations = n_decorations
        if n_decorations > 0:
            self.entity_types.add("d")

        # if d_pos is None or type(d_pos) != list:
        #     d_pos = [d_pos for _ in range(n_decorations)]
        # assert len(d_pos) == n_decorations, f"d_pos should be of length = {n_decorations}"

        self.decorations = [
            Decoration(name=f"Decoration_{i}", size=d_size, pos=d_pos[i],
                       minor_harm_threshold=d_min_harm_thresh, major_harm_threshold=d_maj_harm_thresh)
            for i in range(n_decorations)
        ]

        self.randomise_d_pos = randomise_d_pos
        ###############

        ############
        # LANDMARK #
        ############
        self.landmark = Landmark(name="Landmark", size=lm_size, pos=lm_pos)
        self.randomise_lm_pos = randomise_lm_pos
        self.landmark_found = False
        self.landmark_just_found = False
        ############

        #################
        # -- HAZARDS -- #
        #################
        self.hazards: List[Hazard] = self.pillars + self.lavas
        self.fatal_hazards: List[Hazard] = self.lavas
        # self.hazards: List[Hazard] = list(filter(lambda entity: isinstance(entity, Hazard), self.entities))
        # self.fatal_hazards: List[Hazard] = list(filter(lambda entity: entity.is_fatal, self.hazards))
        #################

        ################
        # ALL ENTITIES #
        ################
        self.entities = [self.robot, self.landmark] + self.humans + self.evil_robots \
                        + self.pillars + self.lavas + self.decorations
        ################

        ##############
        # COLLIDABLE #
        #  ENTITIES  #
        ##############
        self.collidables: List[Entity] = list(filter(lambda entity: entity.collide, self.entities))
        ##############

        #############
        #  MOVABLE  #
        # ENTITIES  #
        #############
        self.movables: List[Entity] = list(filter(lambda entity: entity.movable, self.entities))
        #############

        #############
        #  ACTABLE  #
        # ENTITIES  #
        #############
        self.agents: List[AgentEntity] = [self.robot] + self.evil_robots
        # An agent may stop being actable - e.g. if it is majorly harmed
        self.actables: List[AgentEntity] = list(filter(lambda entity: entity.actable, self.agents))
        #############

        #####################
        # HARMABLE ENTITIES #
        #####################
        self.harmables: List[HarmableEntity] = list(filter(lambda entity: isinstance(entity, HarmableEntity),
                                                           self.entities))
        #####################

        ################
        # -- EVENTS -- #
        ################
        # unique danger state occurrences
        self.events = []
        self.harm_events = []
        # self.danger_occurs = {}
        ################

        self.danger_states = []

        ###############
        # TERMINATION #
        ###############
        self.is_terminated = False
        self.term_reason = None
        # if terminal_states is None:
        #     terminal_states = set()
        # self.terminal_states = terminal_states
        ###############

    # def set_positions(
    #         self,
    #         robot_pos: Optional[np.ndarray] = None,
    #         landmark_pos: Optional[np.ndarray] = None,
    #         human_pos: Optional[List[np.ndarray]] = None,
    #         evil_robot_pos: Optional[List[np.ndarray]] = None,
    #         lava_pos: Optional[List[np.ndarray]] = None,
    #         pillar_pos: Optional[List[np.ndarray]] = None,
    #         decoration_pos: Optional[List[np.ndarray]] = None,
    # ):
    #     pass

    def set_random_pos(
            self,
            entity,
            buffer: SupportsFloat = 0.0,
            max_tries: int = 100
    ):
        x_min = self.x_min + entity.size
        x_max = self.x_max - entity.size

        y_min = self.y_min + entity.size
        y_max = self.y_max - entity.size

        curr_try = 0
        valid_pos = False
        while curr_try < max_tries and not valid_pos:
            entity.pos = self.rng.uniform(low=[y_min, x_min], high=[y_max, x_max], size=(2, )).astype(float)

            valid_pos = True
            for other_entity in self.entities:
                if other_entity is entity:
                    continue

                if self.check_intersection(entity, other_entity, buffer):
                    valid_pos = False
                    break   # Break out of for loop then continue to while loop

            curr_try += 1

        if not valid_pos:
            raise RuntimeError("Max tries reached")

    def set_multi_random_pos(
            self,
            entity_groups: Set[str],  # Which entities to randomise the position of
            buffer: SupportsFloat = 0,  # Ensures buffer around entities
            max_tries: int = 100,  # How many tries for randomising positions
    ):
        # Note: This implementation is more likely to lead to situations where there is no valid position for later
        # entities
        for group in entity_groups:
            if group == "r":  # Robot
                self.set_random_pos(self.robot, buffer, max_tries)
            elif group == "lm":  # Landmark
                self.set_random_pos(self.landmark, buffer, max_tries)
            elif group == "h":  # Humans
                for human in self.humans:
                    self.set_random_pos(human, buffer, max_tries)
            elif group == "er":  # Evil Robots
                for evil_robot in self.evil_robots:
                    self.set_random_pos(evil_robot, buffer, max_tries)
            elif group == "l":  # Lavas
                for lava in self.lavas:
                    self.set_random_pos(lava, buffer, max_tries)
            elif group == "p":  # Pillars
                for pillar in self.pillars:
                    self.set_random_pos(pillar, buffer, max_tries)
            elif group == "d":  # Decorations
                for decoration in self.decorations:
                    self.set_random_pos(decoration, buffer, max_tries)
            else:
                raise ValueError(f"Invalid value for group={group}. "
                                 "Valid values = {'r', 'lm', 'h', 'er', 'l', 'p', 'd'}")
                # raise NotImplementedError(f"Randomise for group={group} not implemented yet")

    # ... #
    def calc_evil_robot_actions(self):
        if len(self.evil_robots) == 0:
            # Do nothing if no evil robots
            return

        for evil_robot in self.evil_robots:
            if evil_robot.actable:
                closest_human = self.get_closest_entity(evil_robot, self.humans)
                # Action taken is unit vector of direction to closest human
                evil_robot.action = self.calc_direction(evil_robot, closest_human)
            else:
                evil_robot.action = None

        # raise NotImplementedError

    # def set_robot_action(self, action: np.ndarray):
    #     assert action.shape == (2, 1)
    #     self.robot.action = action

    ############
    # DYNAMICS #
    ############
    # reset world
    def reset(
            self,
            seed: Optional[int] = None,
            r_pos: Optional[np.ndarray] = None,
            lm_pos: Optional[np.ndarray] = None,
            h_pos: Optional[List[np.ndarray]] = None,
            l_pos: Optional[List[np.ndarray]] = None,
            er_pos: Optional[List[np.ndarray]] = None,
            p_pos: Optional[List[np.ndarray]] = None,
            d_pos: Optional[List[np.ndarray]] = None,
    ):
        """
        Resets the world state by updating positions of specified entities
        such as robot, humans, evil robots, lavas, pillars, decorations, and
        landmark. If specific positions an entity is not provided,
        the function will randomize their positions based on global params.

        :param seed: Optional integer to seed the random number generator for
                     reproducibility
        :param r_pos: Optional numpy array specifying the position of the robot
        :param lm_pos: Optional numpy array specifying the position of the landmark
        :param h_pos: Optional list of numpy arrays specifying the positions of humans
        :param l_pos: Optional list of numpy arrays specifying the positions of lavas
        :param er_pos: Optional list of numpy arrays specifying the positions of evil
                       robots
        :param p_pos: Optional list of numpy arrays specifying the positions of
                      pillars
        :param d_pos: Optional list of numpy arrays specifying the positions of
                      decorations
        :return: None
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._reset_forces()
        self._reset_harms()
        self._reset_events()  # Does this do anything?
        self._reset_velocities()

        self.landmark_found = False

        self.is_terminated = False
        self.term_reason = None

        rand_includes = set()

        #########
        # Robot #
        #########
        if r_pos is not None:
            self.robot.pos = r_pos.astype(float)
        elif self.randomise_r_pos:
            rand_includes.add("r")
        else:
            self.robot.pos = self.robot.init_pos
        #########

        ##########
        # Humans #
        ##########
        if h_pos is not None:
            for i, curr_pos in enumerate(h_pos):
                self.humans[i].pos = curr_pos.astype(float)
        elif self.randomise_h_pos:
            rand_includes.add("h")
        ############
        
        ###############
        # Evil Robots #
        ###############
        if er_pos is not None:
            for i, curr_pos in enumerate(er_pos):
                self.evil_robots[i].pos = curr_pos.astype(float)
        elif self.randomise_er_pos:
            rand_includes.add("er")
        else:
            for evil_robot in self.evil_robots:
                evil_robot.pos = evil_robot.init_pos
        ###############

        #########
        # Lavas #
        #########
        if l_pos is not None:
            for i, curr_pos in enumerate(l_pos):
                self.lavas[i].pos = curr_pos.astype(float)
        elif self.randomise_l_pos:
            rand_includes.add("l")
        #########

        #########
        # Pillars #
        #########
        if p_pos is not None:
            for i, curr_pos in enumerate(p_pos):
                self.pillars[i].pos = curr_pos.astype(float)
        elif self.randomise_p_pos:
            rand_includes.add("p")
        #########

        ##############
        # Decorations #
        ##############
        if d_pos is not None:
            for i, curr_pos in enumerate(d_pos):
                self.decorations[i].pos = curr_pos.astype(float)
        elif self.randomise_d_pos:
            rand_includes.add("d")
        ##############

        ############
        # Landmark #
        ############
        if lm_pos is not None:
            self.landmark.pos = lm_pos.astype(float)
        elif self.randomise_lm_pos:
            rand_includes.add("lm")
        ############


        # Randomise Locations #
        if len(rand_includes) > 0:
            self.set_multi_random_pos(rand_includes)


    # update state of the world
    def step(self, robot_action: np.ndarray):
        self.landmark_just_found = False
        # Reset for when world is terminated
        self.danger_states = []
        self._reset_events()
        self._reset_forces()
        for entity in self.entities:
            if isinstance(entity, HarmableEntity):
                entity.just_harmed = False

        if not self.is_terminated:
            self.robot.action = robot_action
            if self.control_noise > 0:
                self.robot.action = self.rng.normal(loc=robot_action, scale=self.control_noise)

            self.calc_evil_robot_actions()
            self.apply_action_forces()
            self.apply_fatal_collisions()
            self.apply_collision_forces()

            self.calc_harms()
            self.calc_danger_states()

            self.update_states()
            self.handle_walls_hack()

            # # Check if landmark found
            # if self.check_intersection(self.robot, self.landmark):
            #     self.landmark_just_found = True
            #     self.landmark_found = True

            self.comp_termination()

    # RESET #
    def _reset_forces(self):
        for entity in self.entities:
            entity.reset_force()
            # entity.force = np.zeros(2, dtype=float)
            # entity.force_ext = np.zeros(2, dtype=float)
            # entity.force_int = np.zeros(2, dtype=float)
            # if isinstance(entity, HarmableEntity):
            #     entity.force_env = np.zeros(2, dtype=float)
            #     entity.force_robot = np.zeros(2, dtype=float)

    def _reset_velocities(self):
        for entity in self.entities:
            entity.vel = np.zeros(2, dtype=float)

    def _reset_harms(self):
        for entity in self.entities:
            if isinstance(entity, HarmableEntity):
                entity.reset_harm()

    def _reset_events(self):
        self.harm_events = []
        self.events = []
        # self.danger_occurs = {}

    # TODO
    def apply_fatal_collisions(self):
        checked = set()  # Entity pairs which have already been considered

        # Calculate forces applied from collisions between movable entities and collidable entities
        for h_entity in self.harmables:  # harmable entity
            if not (h_entity.movable and h_entity.is_harmable):
                # Skip if h_entity not movable or harmable
                continue

            for f_entity in self.fatal_hazards:  # fatal hazard
                if (h_entity.name, f_entity.name) in checked:
                    continue
                if h_entity is f_entity:
                    # Entity cannot collide with itself
                    # Not sure when a movable entity would be a fatal hazard though
                    continue

                if self.check_intersection(h_entity, f_entity, buffer=-h_entity.size*0.2):
                    # TODO
                    # self.events.append((h_entity, f_entity, Harm.MajorHarm, "fatal_collision"))
                    h_entity.movable = False
                    h_entity.actable = False
                    h_entity.is_harmable = False
                    h_entity.current_harm = Harm.MajorHarm
                    h_entity.just_harmed = True
                    h_entity.harm_source = HarmSource.Env

                checked.add((h_entity.name, f_entity.name))

        # raise NotImplementedError

    # gather agent action forces
    def apply_action_forces(self):
        # Re-calc actables in case it has changed
        self.actables: List[AgentEntity] = list(filter(lambda entity: entity.actable, self.agents))

        for entity in self.actables:
            assert entity.action is not None
            # Apply force based off control signal using speediness as modifier
            entity.force_int = entity.action * entity.speediness  # Assuming correct dimensions & types

    @staticmethod
    def check_intersection(
            entity1: Entity,
            entity2: Entity,
            buffer: SupportsFloat = 0.0
    ) -> bool:
        y1, x1 = entity1.pos[0], entity1.pos[1]
        y2, x2 = entity2.pos[0], entity2.pos[1]
        # Collision if Euclidean distance between centres of two entities is less than the sum of their radiuses plus
        # buffer
        is_collide = (x2 - x1) ** 2 + (y2 - y1) ** 2 < ((entity1.size + entity2.size + buffer)**2)
        return is_collide

    # Return unit vector of direction of entity2 relative to entity1
    @staticmethod
    def calc_direction(entity1: Entity, entity2: Entity) -> np.ndarray:
        delta_pos = entity2.pos - entity1.pos
        norm = np.linalg.norm(delta_pos)
        if norm == 0:
            return np.zeros_like(delta_pos)  # Avoid division by zero, return zero vector
        return delta_pos / norm

    @staticmethod
    def calc_distance_magn(entity1: Entity, entity2: Entity) -> SupportsFloat:
        return np.linalg.norm(entity1.pos - entity2.pos)

    @staticmethod
    def get_closest_entity(entity: Entity, other_entities: List[Entity]) -> Entity:
        if entity in other_entities:
            raise ValueError(f"get_closest_entity not supported if other_entities includes entity")
        distances = [(other_entity, World.calc_distance_magn(entity, other_entity)) for other_entity in other_entities]
        closest_entity = min(distances, key=lambda x: x[1])[0]
        return closest_entity

    def calc_collision_force(self, entity1: Entity, entity2: Entity) -> np.ndarray:
        delta_pos = entity1.pos - entity2.pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # intersection distance
        inters_dist = entity1.size + entity2.size
        if dist <= inters_dist:  # If collision
            # Softmax penetration (taken from MPE)
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - inters_dist) / k) * k
            force = self.contact_force * delta_pos / dist * penetration
        else:
            force = np.zeros(2, dtype=float)

        return force

    # NOTE: This is bugged
    # TODO: FIX
    # gather collision forces
    def apply_collision_forces(self):
        # Re-calc in case they have changed
        self.collidables: List[Entity] = list(filter(lambda entity: entity.collide, self.entities))
        self.movables: List[Entity] = list(filter(lambda entity: entity.movable, self.entities))

        checked = set()  # Entity pairs which have already been considered

        # Calculate forces applied from collisions between movable entities and collidable entities
        for m_entity in self.movables:
            if not m_entity.collide:
                # Skip entity if cannot collide
                continue

            for c_entity in self.collidables:
                # Skip if collision has been calculated
                if (m_entity.name, c_entity.name) in checked or (c_entity.name, m_entity.name) in checked:
                    continue

                # Entity cannot collide with itself - so skip
                if m_entity is c_entity:
                    continue

                # Only calculate collision forces if there is an intersection between entities
                if self.check_intersection(c_entity, m_entity):
                    collide_force = self.calc_collision_force(m_entity, c_entity)
                    m_force = collide_force
                    c_force = -1 * collide_force

                    # # Total force applied
                    # m_entity.force = m_entity.force + m_force
                    # c_entity.force = c_entity.force + c_force

                    # Movable entity
                    if isinstance(m_entity, HarmableEntity):
                        m_entity.force_ext += m_force
                        if c_entity is self.robot:
                            m_entity.force_robot += m_force
                            # m_entity.force_source.add(ForceSorce.Robot)
                        else:
                            m_entity.force_env += m_force
                            # m_entity.force_source.add(ForceSorce.Env)

                    # Collidable entity
                    if isinstance(c_entity, HarmableEntity):
                        c_entity.force_ext += c_force
                        if m_entity is self.robot:
                            c_entity.force_robot += c_force
                            # c_entity.force_source.add(ForceSorce.Robot)
                        else:
                            c_entity.force_env += c_force
                            # c_entity.force_source.add(ForceSorce.Env)

                # if isinstance(c_entity, HarmableEntity):
                #     c_entity.force_ext += c_force
                #
                # if isinstance(c_entity, HarmableEntity):
                #     c_entity.force_ext = c_entity.force_ext + c_force
                #     if m_entity is self.robot:  # If force applied by robot
                #         c_entity.force_agent = c_entity.force_agent + c_force
                #     else:
                #         c_entity.force_env = c_entity.force_env + c_force

                checked.add((m_entity.name, c_entity.name))

            # TODO: Handle wall collisions

    def calc_harms(self):
        # Recalc in case it has changed
        self.harmables = list(filter(lambda entity: isinstance(entity, HarmableEntity) and entity.is_harmable,
                                     self.entities))

        for entity in self.harmables:
            curr_harm = entity.current_harm

            if curr_harm == Harm.MajorHarm:
                continue

            magn_force = np.linalg.norm(entity.force_ext)
            magn_robot_force = np.linalg.norm(entity.force_robot)
            magn_env_force = np.linalg.norm(entity.force_env)

            if magn_force >= entity.major_harm_threshold:
                # entity.current_harm = Harm.MajorHarm

                # Assign blame to robot if it exerted greater force
                if magn_robot_force >= magn_env_force:
                    harm_source = HarmSource.Robot
                else:
                    harm_source = HarmSource.Env

                entity.set_harm(Harm.MajorHarm, harm_source)

                # entity.movable = False
                # entity.is_harmable = False
                # entity.actable = False
                #
                # entity.just_harmed = True

                continue

            if curr_harm != Harm.MinorHarm and magn_force >= entity.minor_harm_threshold:
                # entity.current_harm = Harm.MinorHarm

                # Assign blame to robot if it exerted greater force
                if magn_robot_force >= magn_env_force:
                    harm_source = HarmSource.Robot
                else:
                    harm_source = HarmSource.Env

                entity.set_harm(Harm.MinorHarm, harm_source)
                # entity.just_harmed = True

                continue

    def calc_danger_states(self):
        danger_states = []
        #########
        # ROBOT #
        #########
        if self.robot.just_harmed:
            danger_states.append(DangerState.RobotHarm)
            if self.robot.current_harm == Harm.MajorHarm:
                danger_states.append(DangerState.MajorRobotHarm)
            elif self.robot.current_harm == Harm.MinorHarm:
                danger_states.append(DangerState.MinorRobotHarm)
            else:
                raise RuntimeError("Robot should not be in harm state")

        ##############
        # - HUMANS - #
        ##############
        for human in self.humans:
            if human.just_harmed:
                danger_states.append(DangerState.HumanHarm)
                if human.current_harm == Harm.MajorHarm:
                    danger_states.append(DangerState.MajorHumanHarm)
                    if human.harm_source == HarmSource.Robot:
                        danger_states.append(DangerState.MajorHumanHarmByRobot)
                    elif human.harm_source == HarmSource.Env:
                        danger_states.append(DangerState.MajorHumanHarmByEnv)
                    else:
                        raise RuntimeError(f"Invalid harm source = {human.harm_source} for human={human.name}")

                elif human.current_harm == Harm.MinorHarm:
                    danger_states.append(DangerState.MinorHumanHarm)
                    if human.harm_source == HarmSource.Robot:
                        danger_states.append(DangerState.MinorHumanHarmByRobot)
                    elif human.harm_source == HarmSource.Env:
                        danger_states.append(DangerState.MinorHumanHarmByEnv)
                    else:
                        raise RuntimeError(f"Invalid harm source = {human.harm_source} for human={human.name}")
                else:
                    raise RuntimeError(f"Human={human.name} should not be in harm state")

                # if human.harm_source == HarmSource.Robot:
                #     danger_states.append(DangerState.HumanHarm)
                # elif human.harm_source == HarmSource.Env:
                #     pass
                # else:
                #     raise RuntimeError(f"Human={human.name} should not be in harm state")
        ##############

        ###############
        # DECORATIONS #
        ###############
        for decoration in self.decorations:
            # If robot just harmed a decoration
            if decoration.just_harmed and decoration.harm_source == HarmSource.Robot:
                danger_states.append(DangerState.PropertyHarm)
                if decoration.current_harm == Harm.MajorHarm:
                    danger_states.append(DangerState.MajorPropertyHarm)
                elif decoration.current_harm == Harm.MinorHarm:
                    danger_states.append(DangerState.MinorPropertyHarm)
                else:
                    raise RuntimeError(f"Invalid harm state = {decoration.current_harm} for decoration={decoration.name}")
        ###############
        self.danger_states = danger_states
        # self.danger_states["utility"] = (self.danger_states["option"] + self.danger_states["outcome"]
        #                              + self.danger_states["option-outcome"])
        # for key in ["outcome", "option", "option-outcome"]:
        #     danger_states[key] = list(set(danger_states[key]))  # Remove duplicates - Note: utility can have duplicates
        return danger_states

    # update states of movable entities - i.e. update positions and velocities
    def update_states(self):
        self.movables: List[Entity] = list(filter(lambda entity: entity.movable, self.entities))

        for entity in self.movables:
            entity.pos += entity.vel * self.dt
            entity.vel = entity.vel * (1 - self.damping)

            entity.force = entity.force_int + entity.force_ext

            # Note: Element wise operations on [x, y] np array
            entity.vel += (entity.force / entity.mass) * self.dt

            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.vel[0]) + np.square(entity.vel[1]))

                if speed > entity.max_speed:
                    # entity.vel = (
                    #     entity.vel
                    #     / np.sqrt(
                    #         np.square(entity.vel[0])
                    #         + np.square(entity.vel[1])
                    #     )
                    #     * entity.max_speed
                    # )
                    # Convert to unit vector with entity.vel / speed then times by entity.max_speed
                    # Keep direction but scale vector to have speed of max_speed
                    entity.vel = entity.vel / speed * entity.max_speed

    # Hacky way to get agents to bounce off walls - TODO: Do this the proper way
    def handle_walls_hack(self):
        for entity in self.movables:
            x_min = self.x_min + entity.size
            y_min = self.y_min + entity.size
            x_max = self.x_max - entity.size
            y_max = self.y_max - entity.size

            # If is_walled and agent 'hit' wall then reflect
            y, x = entity.pos[0], entity.pos[1]
            yv, xv = entity.vel[0], entity.vel[1]
            # left wall
            if x <= x_min  and xv < 0:
                xv *= -1
            # right wall
            elif x >= x_max and xv > 0:
                xv *= -1

            # top wall
            if y <= y_min and yv < 0:
                yv *= -1
            # right wall
            elif y >= y_max and yv > 0:
                yv *= -1

            entity.vel[0] = yv
            entity.vel[1] = xv

    def comp_termination(self):
        # term_reason = None
        if not self.is_terminated:
            if self.check_intersection(self.robot, self.landmark):
                self.landmark_just_found = True
                self.landmark_found = True
                self.is_terminated = True
                self.term_reason = "landmark_found"
                return self.is_terminated, self.term_reason

            if self.robot.current_harm == Harm.MajorHarm:
                self.is_terminated = True
                self.term_reason = "major_robot_harm"
                return self.is_terminated, self.term_reason

        return self.is_terminated, self.term_reason
    ############

    def print_forces(self):
        print("# --- FORCES --- #")
        for entity in self.entities:
            print(f"entity = {entity.name}:")
            force_mgn_int = entity.force_magn(entity.force_int)
            force_mgn_ext = entity.force_magn(entity.force_ext)
            print(f"\tforce = {force_mgn_ext + force_mgn_int}:")
            print(f"\tforce_int = {force_mgn_int}:")
            print(f"\tforce_ext = {force_mgn_ext}:")
            if isinstance(entity, HarmableEntity):
                print(f"\tforce_env = {entity.force_magn(entity.force_env)}")
                print(f"\tforce_robot = {entity.force_magn(entity.force_robot)}")
            print()
        print("# -------------- #")

