import os
from typing import Union, Optional, Dict, List

import numpy as np
import pygame

from morality_gym.environments.core.entity.base import BaseEntity
from morality_gym.environments.core.state import WorldState
from morality_gym.environments.trolley.entity import RailSwitch
from morality_gym.utils.common import join_paths, comp_base_path


# AI Edited Class - May be sketchy
class EntitySprite(pygame.sprite.Sprite):
    def __init__(
            self,
            entity: BaseEntity,
            tile_size: int,
            tile_surfaces: Dict[str, pygame.Surface],
            font: Optional[pygame.font.Font] = None, # Optional: Pass font for better practice
            cross_offset: int = 4 # Offset for the red cross
    ):
        super().__init__()

        self.entity = entity
        self.tile_size = tile_size
        self.tile_surfaces = tile_surfaces
        self.cross_offset = cross_offset # Store the offset

        self.state_to_asset_fn = entity.to_asset_fn
        self.state_to_rot_fn = entity.to_rot_fn
        self.state_to_alpha_fn = entity.to_alpha_fn

        self.is_vis_amount = entity.is_vis_amount

        # --- Font Handling ---
        if font is None:
            if not pygame.font.get_init():
                 pygame.font.init()
            self.font = pygame.font.Font(None, int(tile_size * 0.5)) # Smaller font size
        else:
            self.font = font
        # --- End Font Handling ---

        if self.state_to_asset_fn is None:
            self.state_to_asset_fn = lambda x: "default"

        self.curr_rot: Optional[int] = None
        self.curr_asset: Optional[str] = None
        self.curr_alpha: Optional[int] = None

        self.rect: pygame.Rect = pygame.Rect(0, 0, self.tile_size, self.tile_size)
        self.original_image: Optional[pygame.Surface] = None
        self.image: Optional[pygame.Surface] = None # This is the final image drawn

        self._layer = entity.vis_layer
        self.update()


    def _update_pos(self):
        if self.entity.pos is None:
            raise ValueError(f"Entity {self.entity.name} has no pos.")

        y, x = self.entity.pos
        self.rect.x = (x + 1) * self.tile_size
        self.rect.y = (y + 1) * self.tile_size

    def _render_amount(self):
        """Renders the entity's amount in a circle onto the sprite's image."""
        # This method now assumes self.image is already a copy of original_image
        # It modifies self.image directly.
        if self.image is None:
             return # Should not happen if _update_image runs first

        if self.entity.amount is not None: # Render if amount is not None (includes 1)
            amount_text = str(self.entity.amount)
            text_surface = self.font.render(amount_text, True, (0, 0, 0)) # Black text
            text_rect = text_surface.get_rect()

            center_x, center_y = self.image.get_rect().center
            radius = max(text_rect.width // 2, text_rect.height // 2) + int(self.tile_size * 0.05)
            radius = max(radius, int(self.tile_size * 0.15))
            outline_thickness = 1

            pygame.draw.circle(self.image, (255, 255, 255), (center_x, center_y), radius)
            pygame.draw.circle(self.image, (0, 0, 0), (center_x, center_y), radius, outline_thickness)

            text_rect.center = (center_x, center_y)
            self.image.blit(text_surface, text_rect)


    def _update_image(self):
        """Updates the sprite's visual representation based on entity state."""
        asset_name = self.state_to_asset_fn(self.entity)
        if type(asset_name) == list:
            raise NotImplementedError("List of assets not supported yet.")

        if asset_name not in self.tile_surfaces:
             print(f"Warning: Asset '{asset_name}' not found for entity '{self.entity.name}'. Using 'default'.")
             asset_name = "default"
             if "default" not in self.tile_surfaces:
                  raise ValueError("Default asset 'default' not found in tile_surfaces.")

        # Start with the base image for this asset
        base_image = self.tile_surfaces[asset_name].copy()

        # Apply rotation if needed
        if self.state_to_rot_fn is not None:
            if isinstance(self.entity, RailSwitch):
                pass
            rot = self.state_to_rot_fn(self.entity)
            if type(rot) == list:
                raise NotImplementedError("List of rotations not supported yet.")
            base_image = pygame.transform.rotate(base_image, rot)

        # Set the potentially rotated image as the current image
        # Amount and harm visuals will be drawn on top of this
        self.image = base_image

        # Render amount text/circle if required (modifies self.image)
        if self.is_vis_amount:
            self._render_amount()

        # --- Draw Red Cross if Harmed ---
        if self.entity.is_harmed: # Check the is_harmed property
            RED = (255, 0, 0)
            line_thickness = 2 # Adjust thickness as needed
            size = self.tile_size
            offset = self.cross_offset

            # Ensure offset doesn't make lines disappear or invert
            offset = max(0, min(size // 2 - line_thickness, offset))

            # Line 1: Top-left to Bottom-right
            start_pos1 = (offset, offset)
            end_pos1 = (size - 1 - offset, size - 1 - offset) # Use size-1 for 0-based index
            pygame.draw.line(self.image, RED, start_pos1, end_pos1, line_thickness)

            # Line 2: Top-right to Bottom-left
            start_pos2 = (size - 1 - offset, offset)
            end_pos2 = (offset, size - 1 - offset)
            pygame.draw.line(self.image, RED, start_pos2, end_pos2, line_thickness)
        # --- End Harm Visual ---

        # Apply alpha transparency *last* to the final image
        if self.state_to_alpha_fn is not None:
            alpha = self.state_to_alpha_fn(self.entity)
            if type(alpha) == list:
                raise NotImplementedError("List of alphas not supported yet.")
            alpha = max(0, min(255, int(alpha)))
            self.image.set_alpha(alpha)

    def update(self, *args, **kwargs):
        """Updates the sprite's position and visual appearance."""
        self._update_pos()
        self._update_image()



# def load_asset_image(
#         asset_name: str,
#         tile_size: int,
#         convert_surface: pygame.Surface
# ) -> pygame.Surface:
#     image_path = join_paths([ASSETS_PATH, f"{asset_name}.png"])
#     img = pygame.image.load(image_path).convert_alpha(convert_surface)
#     img = pygame.transform.scale(img, (tile_size, tile_size))
#     return img

class Renderer:
    VALID_RENDER_MODES = {"human", "rgb_array", "disabled"}
    # VALID_ASSETS = {"robot", "floor", "wall_1", "wall_2", "wall_3"}
    def __init__(
            self,
            world_state: WorldState,
            # entity_state_to_asset_fns: Dict[str, Callable[[BaseEntity], str]],
            # entity_state_to_rot_fns: Dict[str, Union[Callable[[BaseEntity], int], None]],
            # entity_layers: Dict[str, int],
            # window_width: int = 1000,
            window_height: int = 700,
            grid_border: int = 16,
            text_box_width: int = 1,
            text_box_margin: int = 1,
            render_mode: str = "human",
            asset_base_paths: Optional[Union[str, List[str]]] = None
    ):
        self.world_state = world_state

        self.render_mode = render_mode
        self.enabled = render_mode != "disabled"

        # self.window_width = window_width
        self.window_height = window_height
        self.grid_border = grid_border
        self.text_box_width = text_box_width
        self.text_box_margin = text_box_margin

        # CALC STUFF - TODO: Better Comment :)
        # avail_grid_width_px = self.window_width - 2 * self.grid_border - self.text_box_width
        avail_grid_height_px = self.window_height - 2 * self.grid_border
        # min_avail_grid_size_px = min(avail_grid_width_px, avail_grid_height_px)
        min_avail_grid_size_px = avail_grid_height_px

        self.grid_width = self.world_state.grid_width + 2  # Account for walls
        self.grid_height = self.world_state.grid_height + 2  # Account for walls
        # max_grid_dim = max(self.grid_width, self.grid_height)
        max_grid_dim = self.grid_height

        self.tile_size = min_avail_grid_size_px // max_grid_dim
        self.grid_width_px = self.grid_width * self.tile_size
        self.grid_height_px = self.grid_height * self.tile_size
        self.grid_tl = (self.grid_border, self.grid_border)

        self.window_width = self.grid_width_px + 2 * self.grid_border + self.text_box_width
        # END CALC STUFF

        self.window = None
        self.clock = None

        self._is_init = False

        self._init_render()

        self._display_surface = pygame.Surface((self.grid_width_px, self.grid_height_px)) #.convert_alpha()
        if render_mode == "human":
            self._display_surface = self._display_surface.convert_alpha()

        ####################
        # GET VALID ASSETS #
        ####################
        if asset_base_paths is None:
            asset_base_paths = []
        elif type(asset_base_paths) == str:
            asset_base_paths = [asset_base_paths]
        elif type(asset_base_paths) != list:
            raise TypeError(f"asset_base_paths must be None, str, or list. Got {type(asset_base_paths)}")

        env_base_path = comp_base_path(os.path.abspath(__file__), level=2)
        for folder_name in ["core", "trolley"]:
            curr_base_path = join_paths([env_base_path, folder_name, "assets"])
            asset_base_paths.append(curr_base_path)

        self.asset_to_path = {}
        self.valid_assets = []
        for curr_base_path in asset_base_paths:
            if not os.path.exists(curr_base_path):
                raise ValueError(f"curr_base_path={curr_base_path} does not exist.")

            curr_asset_files = os.listdir(curr_base_path)
            curr_asset_names = [asset_file.replace(".png", "") for asset_file in curr_asset_files
                                if ".png" in asset_file]
            self.valid_assets.extend(curr_asset_names)

            for asset_name in curr_asset_names:
                self.asset_to_path[asset_name] = join_paths([curr_base_path, f"{asset_name}.png"])

        # asset_files = os.listdir(ASSETS_PATH)
        # self.valid_assets = [asset_file.replace(".png", "") for asset_file in asset_files
        #                      if ".png" in asset_file]

        ####################
        self._background_surface = self._generate_background()

        ###############
        # - SPRITES - #
        ###############
        self.tile_surfaces = {asset_name: self._load_asset_image(asset_name) for asset_name in self.valid_assets}
        entities = world_state.entities
        entity_sprites = []
        for entity_name, entity in entities.items():
            # if entity_name not in entity_state_to_asset_fns:
            #     raise ValueError(f"entity_name={entity_name} not in entity_state_to_asset_fns")
            # if entity_name not in entity_state_to_rot_fns:
            #     raise ValueError(f"entity_name={entity_name} not in entity_state_to_rot_fns")
            # if entity_name not in entity_layers:
            #     raise ValueError(f"entity_name={entity_name} not in entity_layers")

            entity_sprite = EntitySprite(
                entity, self.tile_size, self.tile_surfaces,
                # entity_state_to_asset_fns[entity_name],
                # entity_state_to_rot_fns[entity_name], layer=entity_layers[entity_name]
            )

            entity_sprites.append(entity_sprite)

        self._layered_updates_sprites = pygame.sprite.LayeredUpdates(*entity_sprites)

        ###############


    def _load_asset_image(self, asset_name: str) -> pygame.Surface:
        # image_path = join_paths([ASSETS_PATH, f"{asset_name}.png"])
        image_path = self.asset_to_path[asset_name]
        img = pygame.image.load(image_path).convert_alpha(self._display_surface)
        img = pygame.transform.scale(img, (self.tile_size, self.tile_size))
        return img


    def _generate_background(self) -> pygame.Surface:
        background_surface = pygame.Surface((self.grid_width_px, self.grid_height_px)).convert_alpha(self._display_surface)
        background_surface.fill((0, 0, 0, 0))

        trav_grids_d = self.world_state.traversability_grids
        tmp_arrs = np.array([arr.astype(int) * (ind+1) for ind, arr in trav_grids_d.items()])
        trav_tiles_arr = tmp_arrs.max(axis=0)
        pad_arr = np.ones((trav_tiles_arr.shape[0] + 2, trav_tiles_arr.shape[1] + 2), dtype=int)
        pad_arr[1:-1, 1:-1] = trav_tiles_arr.copy()
        trav_tiles_arr = pad_arr

        val_to_img = {
            0: self._load_asset_image("floor"),
            1: self._load_asset_image("wall_1"),
            2: self._load_asset_image("wall_2"),
            3: self._load_asset_image("wall_3"),
        }

        for y in range(trav_tiles_arr.shape[0]):
            for x in range(trav_tiles_arr.shape[1]):
                tile_val = int(trav_tiles_arr[y, x])
                tile_img = val_to_img[tile_val]
                background_surface.blit(tile_img, (x * self.tile_size, y * self.tile_size))

        return background_surface


    def _init_render(self):
        # Called on first render if not disabled

        if not self.enabled:
            return

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Morality Gym")
            # w, h = self.tot_window_width, self.tot_window_height
            # self.font = pygame.font.Font(None, self.font_size)
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        self._is_init = True

    def _render_frame(self) -> Union[None, np.ndarray]:
        self._display_surface.fill((0, 0, 0, 0))
        self._display_surface.blit(self._background_surface, (0,0))
        self._layered_updates_sprites.update()
        self._layered_updates_sprites.draw(self._display_surface)

        if self.render_mode=="human":
            # self.window.blit(self._background_surface, self.grid_tl)
            self.window.blit(self._display_surface, self.grid_tl)
            pygame.event.pump()
            pygame.display.flip()
            # pygame.display.flip()
            # self.clock.tick(60)

        rgb_arr = np.transpose(np.array(pygame.surfarray.pixels3d(self._display_surface)), axes=(1, 0, 2))
        return rgb_arr

    def render(self) -> Union[None, np.ndarray]:
        if not self.enabled:
            return

        if not self._is_init:
            self._init_render()

        return self._render_frame()

    def close(self):
        if self._is_init:
            if self.render_mode == "human" or self.render_mode == "rgb_array":
                pygame.display.quit()
            pygame.quit()
            self._is_init = False

def main():
    pass

if __name__ == "__main__":
    main()