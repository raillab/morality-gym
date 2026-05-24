# TODO
from morality_gym._environments.core.dynamics.base import BaseDynamics
from morality_gym._environments.core.state import WorldState
from morality_gym._environments.trolley.entity import Railway


class RailwayDynamics(BaseDynamics):
    def __init__(
            self,
            world_state: WorldState
    ):
        super().__init__(world_state)
        if "railway" not in self._world_state.entity_groups:
            raise ValueError(f"railway not found in world_state.entity_groups")
        self.railway: Railway = self._world_state.entity_groups["railway"]


    def _handle_trolleys_enabled(self):
        ws = self._world_state
        if ws.landmark_just_found:
            self.railway.enable_trolleys()

    def __call__(self):
        self._handle_trolleys_enabled()

