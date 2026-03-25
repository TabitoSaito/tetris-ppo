from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from utils.env_helper import generate_unique_coordinates, Identifier


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class FloorPainterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_obstacles=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.num_obstacles = num_obstacles

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "board": spaces.Box(-1, 1, shape=(size, size), dtype=int)
            }
        )

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        self.board = np.zeros((size, size), dtype=int)
        self.vis_board = np.zeros((size, size), dtype=int)

        self.obstacles = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "board": self.board}

    def _get_info(self):
        return {"vis_board": self.vis_board}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # self._agent_location = np.zeros((2,), dtype=int) # test

        self.board = np.zeros((self.size, self.size), dtype=int)
        self.vis_board = np.zeros((self.size, self.size), dtype=int)

        self.obstacles = generate_unique_coordinates(
            self.num_obstacles,
            self.size - 1, 
            self.size - 1, 
            except_=self._agent_location, 
            rng=self.np_random.integers
        )

        # self.obstacles = [[0, 1, 2, 3, 4], [3, 2, 4, 1, 0]] # test

        self.board[*self.obstacles] = Identifier.OBSTACLE.value
        self.vis_board[*self.obstacles] = Identifier.OBSTACLE.value

        self.board[*self._agent_location] = Identifier.PAINTED.value
        self.vis_board[*self._agent_location] += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = 0
        direction = self._action_to_direction[action]

        if not np.any(np.all(np.array(list(zip(*self.obstacles))) == (self._agent_location + direction), axis=1)):
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )

        if self.board[*self._agent_location] == 0:
            reward += 1
            self.board[*self._agent_location] = Identifier.PAINTED.value
        else:
            reward += -0.1

        self.vis_board[*self._agent_location] += 1

        terminated = not np.any(self.board == 0)
        reward += 10 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pygame.font.init()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        my_font = pygame.font.SysFont('Comic Sans MS', round(pix_square_size * 0.9))

        # Draw obstacles
        for obstacle in tuple(zip(*self.obstacles)):
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    obstacle[0] * pix_square_size,
                    obstacle[1] * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                )
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Add visit count
        for x in range(self.size):
            for y in range(self.size):
                visit_count = self.vis_board[x, y]
                text_surf = my_font.render(str(visit_count), False, (0, 0, 0))
                canvas.blit(
                    text_surf,
                    (pix_square_size * x, pix_square_size * y)
                )

        # Add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()