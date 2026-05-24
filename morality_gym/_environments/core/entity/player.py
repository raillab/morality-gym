from typing import Dict, Any

from morality_gym._environments.core.entity.base import BaseEntity


#################
# PLAYER ENTITY #
#################
class PlayerEntity(BaseEntity):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
            character_type="robot",
            has_pre_step=True,
            has_post_step=True
        )
        self._is_player = True

    def calc_obs(self, is_normalise: bool = False) -> Dict[str, Any]:
        obs = {
            "pos": self.calc_pos_obs(is_normalise),
            "is_harmed": self.is_harmed,
            "is_terminated": self.is_terminated
        }

        return obs

    def post_step(self):
        super().post_step()
        if self.is_terminated:
            self.is_actable = False
            self.is_movable = False