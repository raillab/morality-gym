from typing import Optional, Dict, Any

import numpy as np

from morality_gym.environments.core.entity.base import BaseEntity

# TODO (MAYBE): Rename to 'Character'

####################
# - HUMAN ENTITY - #
####################
class CharacterEntity(BaseEntity):
    def __init__(
            self,
            *args,
            # traits: Optional[Dict[str, Any]] = None,
            # is_vis_amount=True,
            **kwargs
    ):
        """Initializes a CharacterEntity instance.

        Args:
            *args: Positional arguments passed directly to the BaseEntity
                   constructor. See BaseEntity.__init__ for details.
            traits: An optional dictionary to store arbitrary key-value pairs
                    representing characteristics of the human (e.g., 'age': 30,
                    'role': 'doctor'). Defaults to None.
            **kwargs: Keyword arguments passed directly to the BaseEntity
                      constructor. See BaseEntity.__init__ for details (common
                      args include name, group, pos, is_collidable, etc.).
        """
        def to_asset_fn(entity: BaseEntity) -> str:
            if self.character_type == "human":
                return "human"
            elif self.character_type == "robot":
                return "robot"
            elif self.character_type == "animal":
                # raise NotImplementedError
                return "animal"
            else:
                raise ValueError(f"Unknown character type: {self.character_type}")
        def to_rot_fn(entity: BaseEntity) -> int:
            return 0
        def to_alpha_fn(entity: BaseEntity) -> int:
            return 255

        super().__init__(
            *args,
            to_asset_fn=to_asset_fn,
            to_rot_fn=to_rot_fn,
            to_alpha_fn=to_alpha_fn,
            has_pre_step=True,
            has_post_step=True,
            is_vis_amount=True,
            **kwargs
        )
        # if traits is None:
        #     traits = {}
        # self.traits = traits

    def calc_obs(self, is_normalise: bool = False) -> Dict[str, Any]:
        obs = {
            "pos": self.calc_pos_obs(is_normalise),
            "is_harmed": self.is_harmed,
            "amount": self.amount
        }
        if self.character_type is not None:
            n_character_types = len(self.valid_character_types)
            onehot_arr = np.zeros(n_character_types)
            ind = self.valid_character_types.index(self.character_type)
            onehot_arr[ind] = 1
            obs["character_type"] = onehot_arr

        return obs


######################
# MULTI HUMAN ENTITY #
######################
# TODO: ...
######################
