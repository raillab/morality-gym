from morality_gym._environments.core.entity.base import BaseEntity

def make_player_fns(base_name):
    def to_asset_fn(entity: BaseEntity) -> str:
        return f"{base_name}"

    def to_rot_fn(entity: BaseEntity) -> int:
        return 0

    def to_alpha_fn(entity: BaseEntity) -> int:
        return 255

    return to_asset_fn, to_rot_fn, to_alpha_fn