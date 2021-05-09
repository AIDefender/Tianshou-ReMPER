#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.wrappers import *
import matplotlib.pyplot as plt


class MazeEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, grid_size=19, max_steps=1000, U_shape=False):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=max_steps)
        self.action_space = spaces.Discrete(3)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.grid.vert_wall(2,1,2)
        self.grid.horz_wall(1,4,4)
        self.grid.horz_wall(6,4,2)
        self.grid.vert_wall(2,6,2)
        self.grid.vert_wall(4,2,5)
        self.grid.horz_wall(4,2,3)
        self.grid.horz_wall(5,6,2)
        # self.grid.vert_wall(6,5,3)
        # self.grid.horz_wall(2,8,5)
        # self.grid.vert_wall(8,6,4)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


def test():
    env = MazeEnv(agent_pos=(1,1), goal_pos=(1,7), grid_size=9, U_shape=True)
    # env = gym.make("MiniGrid-Empty-8x8-v0")
    # env = ViewSizeWrapper(env, agent_view_size=21)
    env = RGBImgObsWrapper(env)
    # env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    s = env.reset()
    print(s.shape)
    plt.imshow(s) 
    plt.savefig("a.png")

if __name__ == '__main__':
    test()
