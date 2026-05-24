from morality_gym._environments.core.action import ActionEnum
from morality_gym._environments.core.dynamics.base import BaseDynamics
from morality_gym._environments.core.state import WorldState


class LandmarkDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState,
            landmark_reached_mode: str = "interact"  # "interact" or "enter"
    ):
        super().__init__(world_state)
        self._player = self._world_state.player
        self._landmark = self._world_state.landmark
        self._is_enabled = self._landmark is not None
        self._reached_on_interact = landmark_reached_mode == "interact"

    def __call__(self):
        if self._is_enabled \
                and (not self._world_state.is_landmark_found and self._player.pos == self._landmark.pos):
            # self._reached_on_interact = True then agent just needs to intersect with landmark
            if not self._reached_on_interact or (self._player.action == ActionEnum.INTERACT):
                self._player.is_terminated = True
                self._world_state.just_terminated.add(self._player.name)
                self._world_state.landmark_just_found = True
                self._world_state.is_landmark_found = True
