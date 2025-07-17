import gym
from griddly import *
from griddly.util.render_tools import RenderToVideo
from griddly import GymWrapperFactory, gd
from gym.wrappers import RecordVideo
from griddly.gym import GymWrapper
from gym.vector import SyncVectorEnv
import random
import copy
import numpy as np

class Trainer():

    def __init__(self):
        self.episodes_total = 0

    def createEnv(self, env_id):
        return GymWrapper(
                env_id,
                player_observer_type=gd.ObserverType.VECTOR,
                global_observer_type=gd.ObserverType.VECTOR, 
                max_steps=200,
        )
    
    def train(self, env_id, agent, episode_length, eval_at, upd_social=False):
        env = self.createEnv(env_id)

        episodes_length = []
        episodes_rewards = []
        episodes_collisions_humanoid = []
        episodes_collisions_wall = []
        for episode in range(episode_length):
            if self.episodes_total in eval_at:
                eval = True
            else:
                eval = False
            obs = env.reset()
            if np.random.uniform(0, 1) <= 0.5:
                reverse = False
            else:
                reverse = True
            episodes_length_0, episodes_rewards_0, collisions_humanoid, collisions_wall = agent.act(env, obs[0][0], eval, reverse, upd_social)
            episodes_length.append(episodes_length_0)
            episodes_rewards.append(episodes_rewards_0)
            episodes_collisions_humanoid.append(collisions_humanoid)
            episodes_collisions_wall.append(collisions_wall)
            self.episodes_total += 1
        return episodes_length, episodes_rewards, episodes_collisions_humanoid, episodes_collisions_wall