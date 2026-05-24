# TODO
from morality_gym._environments.core.custom_types import PosType
from morality_gym._environments.core.entity.base import BaseEntity


class LandmarkEntity(BaseEntity):
    def __init__(
            self,
            name: str,
            group: str,
            pos: PosType,
            vis_layer: int = 1,
    ):
        def to_asset_fn(entity: BaseEntity) -> str:
            return "landmark"

        super().__init__(
            name,
            group,
            pos,
            is_collidable=False, is_intersectable=True,
            is_movable=False, is_actable=False,
            is_interactable=False, is_agent_interactable=False,
            is_scripted=False, is_harmable=False,
            is_terminatable=False,
            to_asset_fn=to_asset_fn, vis_layer=vis_layer,
            has_post_step=False, has_pre_step=False
        )