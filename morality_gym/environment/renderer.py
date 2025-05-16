from typing import Union, Tuple, Optional

import numpy as np
import pygame

from morality_gym.environment.world import World


class Renderer:
    VALID_RENDER_MODES = {"human", "rgb_array", "disabled"}

    def __init__(
            self,
            world: World,
            render_mode: str = "human",
            window_size: int = 750,
            margin_width: int = 10,
            text_box_width: int = 350,
            text_box_margin: int = 10,
            font_size: int = 24,
            render_wait: int = 250,
            fps: int = 16,
            r_color: Union[str, Tuple[int, int, int]] = "blue",
            lm_color: Union[str, Tuple[int, int, int]] = "green",
            h_color: Union[str, Tuple[int, int, int]] = "orange",
            er_color: Union[str, Tuple[int, int, int]] = "black",
            l_color: Union[str, Tuple[int, int, int]] = "red",
            p_color: Union[str, Tuple[int, int, int]] = "gray",
            d_color: Union[str, Tuple[int, int, int]] = "purple",
    ):
        self.world = world

        if render_mode not in self.VALID_RENDER_MODES:
            raise ValueError(f"Invalid value for render_mode = {render_mode}. Valid values = {self.VALID_RENDER_MODES}")
        self.render_mode = render_mode
        self.is_disabled = self.render_mode == "disabled"

        self.y_min, self.y_max = world.y_min, world.y_max
        self.x_min, self.x_max = world.x_min, world.x_max

        self.grid_height = self.y_max - self.y_min
        self.grid_width = self.x_max - self.x_min

        self.margin_width = margin_width
        self.vis_window_size = window_size  # Excluding text box
        self.tot_window_width = self.vis_window_size + text_box_width
        self.tot_window_height = self.vis_window_size

        self.max_grid_dim = max(self.grid_width, self.grid_height)
        self.scale_grid = (self.vis_window_size - 2 * self.margin_width) / self.max_grid_dim
        self.grid_canvas_width = self.grid_width * self.scale_grid
        self.grid_canvas_height = self.grid_height * self.scale_grid

        self.text_box_width = text_box_width
        self.text_box_margin = text_box_margin
        # self.textbox_lt = (self.window_size - self.text_box_margin, self.window_size + self.margin_width)

        self.font = None
        self.font_size = font_size
        # self.robot_color = robot_color
        # self.landmark_color = landmark_color
        # self.lava_color = lava_color

        self.r_color = r_color
        self.lm_color = lm_color
        self.er_color = er_color
        self.h_color = h_color
        self.p_color = p_color
        self.d_color = d_color
        self.l_color = l_color

        self.render_wait = render_wait
        self.fps = fps

        self.clock = None
        self.window = None

        self._is_init = False

    def _init(self):
        if self.is_disabled:
            raise ValueError

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            w, h = self.tot_window_width, self.tot_window_height
            self.font = pygame.font.Font(None, self.font_size)
            self.window = pygame.display.set_mode((w, h))

            self.clock = pygame.time.Clock()

        self._is_init = True

    def _get_canvas_coord(self, grid_x, grid_y):
        margin_width = self.margin_width
        scale = self.scale_grid
        x = scale * (grid_x - self.x_min) + margin_width
        y = self.vis_window_size - scale * (grid_y - self.y_min) - margin_width
        return int(x), int(y)

    @staticmethod
    def _draw_circle(
            canvas,
            x, y,
            radius,
            fill_color,
            line_color="black",
            line_width=1
    ):
        pygame.draw.circle(canvas, line_color, (x, y), radius, width=line_width)
        pygame.draw.circle(canvas, fill_color, (x, y), radius)

    def _draw_entity(
            self,
            canvas,
            entity,
            fill_color,
            line_color="black",
            line_width=1
    ):
        gx, gy = entity.pos[1], entity.pos[0]
        x, y = self._get_canvas_coord(gx, gy)
        r = entity.size * self.scale_grid
        self._draw_circle(canvas, x, y, r, fill_color, line_color, line_width)

    # def _draw_textbox(
    #         self,
    #         text:str,
    #         text_color="black"
    # ):
    #

    def _draw_textbox(
            self,
            canvas,
            text: str,
            color="black"
    ):
        font = self.font

        x = self.vis_window_size + self.text_box_margin
        y = self.vis_window_size - self.text_box_margin

        width, height = (self.text_box_width - self.text_box_margin * 2, self.vis_window_size - 2 * self.text_box_margin)
        rect = pygame.Rect(x, y, width, height)

        pygame.draw.rect(canvas, "white", rect)

        lines = text.split("\n")
        line_height = font.get_linesize()

        y = self.text_box_margin

        y_offset = 0
        for line in lines:
            text_surface = font.render(line, True, color)

            canvas.blit(text_surface, (x, y + y_offset))
            # canvas.blit(text_surface, (x, 10))
            y_offset += line_height

            # Basic clipping: Stop drawing if we reach the bottom of the rect
            if y + y_offset + line_height > y + height:
                break  # or add scrolling logic here

        # pygame.draw.rect(window, "white", rect)

    def _render_world_details(self, canvas):
        text = ""
        world = self.world

        #########
        # ROBOT #
        #########
        robot = world.robot
        text += "# ROBOT #\n"
        text += f"pos: {robot.pos.round(2)}; vel: {robot.vel.round(2)}; speed: {round(np.linalg.norm(robot.vel), 2)} \n"
        r_force = robot.force.round(2)
        r_force_ext = robot.force_ext.round(2)
        r_force_int = robot.force_int.round(2)
        text += f"force -> total: {r_force} (magn = {round(np.linalg.norm(r_force), 2)})\n"
        text += f"         ext: {r_force_ext} (magn = {round(np.linalg.norm(r_force_ext), 2)})\n"
        text += f"         int: {r_force_int} (magn = {round(np.linalg.norm(r_force_int), 2)})\n"
        text += f"curr_harm: {robot.current_harm.name} \n"
        # text += f"max_speed: {world.robot.max_speed}"
        ##########
        # HUMANS #
        ##########
        # if world.n_humans > 0:
        #     text += "# HUMAN #\n"
        for human in world.humans:
            text += f"\n# HUMAN - {human.name} #\n"
            h_force = human.force.round(2)
            h_force_ext = human.force_ext.round(2)
            h_force_int = human.force_int.round(2)
            text += f"force -> total: {h_force} (magn = {round(np.linalg.norm(h_force), 2)})\n"
            text += f"         ext: {h_force_ext} (magn = {round(np.linalg.norm(h_force_ext), 2)})\n"
            text += f"         int: {h_force_int} (magn = {round(np.linalg.norm(h_force_int), 2)})\n"
            text += f"curr_harm: {human.current_harm.name} \n"

        ###############
        # DECORATIONS #
        ###############
        for decoration in world.decorations:
            text += f"\n# DECORATION - {decoration.name} #\n"
            d_force = decoration.force.round(2)
            d_force_ext = decoration.force_ext.round(2)
            d_force_int = decoration.force_int.round(2)
            text += f"force -> total: {d_force} (magn = {round(np.linalg.norm(d_force), 2)})\n"
            text += f"         ext: {d_force_ext} (magn = {round(np.linalg.norm(d_force_ext), 2)})\n"
            text += f"         int: {d_force_int} (magn = {round(np.linalg.norm(d_force_int, 2))})\n"
            text += f"curr_harm: {decoration.current_harm.name} \n"

        ###############
        # EVIL ROBOTS #
        ###############
        # if world.n_humans > 0:
        #     text += "# HUMAN #\n"
        for evil_robot in world.evil_robots:
            text += f"\n# EVIL ROBOT - {evil_robot.name} #\n"
            text += f"pos: {evil_robot.pos.round(2)}; vel: {evil_robot.vel.round(2)}; speed: {round(np.linalg.norm(evil_robot.vel), 2)} \n"
            er_force = evil_robot.force.round(2)
            er_force_ext = evil_robot.force_ext.round(2)
            er_force_int = evil_robot.force_int.round(2)
            text += f"force -> total: {er_force} (magn = {round(np.linalg.norm(er_force), 2)})\n"
            text += f"         ext: {er_force_ext} (magn = {round(np.linalg.norm(er_force_ext), 2)})\n"
            text += f"         int: {er_force_int} (magn = {round(np.linalg.norm(er_force_int), 2)})\n"
            text += f"curr_harm: {evil_robot.current_harm.name} \n"


        # text += f"vel: {self.world.robot.vel} \n"

        self._draw_textbox(canvas, text)



    def _render_frame(
            self,
            render_wait: Optional[int] = None
    ) -> np.ndarray:
        if render_wait is None:
            render_wait = self.render_wait

        world=self.world
        # window_size = self.window_size
        scale = self.scale_grid
        margin = self.margin_width

        canvas = pygame.Surface((self.tot_window_width, self.tot_window_height))
        canvas.fill((255, 255, 255))  # Fill background as white

        ##############
        # GRID LINES #
        ##############
        # Outer border
        rect_border = pygame.Rect(margin, self.vis_window_size - margin - self.grid_canvas_height,
                                  self.grid_canvas_width, self.grid_canvas_height)
        pygame.draw.rect(canvas, "black", rect_border, 1)

        # Origin Lines
        # X
        pygame.draw.line(canvas, "black", self._get_canvas_coord(self.x_min, 0),
                         self._get_canvas_coord(self.x_max, 0), 2)
        # Y
        pygame.draw.line(canvas, "black", self._get_canvas_coord(0, self.y_min),
                         self._get_canvas_coord(0, self.y_max), 2)
        ##############

        ############
        # LANDMARK #
        self._draw_entity(canvas, world.landmark, self.lm_color, line_color="black", line_width=1)
        ############

        #############
        # - LAVAS - #
        for lava in world.lavas:
            self._draw_entity(canvas, lava, self.l_color, line_color="black", line_width=1)
        #############

        ###########
        # PILLARS #
        ###########
        for pillar in world.pillars:
            self._draw_entity(canvas, pillar, self.p_color, line_color="black", line_width=1)
        ###########

        ###############
        # DECORATIONS #
        ###############
        for decoration in world.decorations:
            self._draw_entity(canvas, decoration, self.d_color, line_color="black", line_width=1)
        ###############

        ##############
        # - HUMANS - #
        for human in world.humans:
            self._draw_entity(canvas, human, self.h_color, line_color="black", line_width=1)
        ##############

        ###############
        # EVIL ROBOTS #
        ###############
        for er in world.evil_robots:
            self._draw_entity(canvas, er, self.er_color, line_color="black", line_width=1)
        ###############

        #############
        # - ROBOT - #
        self._draw_entity(canvas, world.robot, self.r_color, line_color="black", line_width=1)
        #############


        if self.render_mode == "human":
            self._render_world_details(canvas)
            self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()
            pygame.time.wait(render_wait)

        rgb_arr = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        return rgb_arr

    def render(
            self,
            render_wait: Optional[int] = None
    ) -> Union[None, np.ndarray]:
        if self.is_disabled:
            return

        if not self._is_init:
            self._init()

        return self._render_frame(render_wait)

