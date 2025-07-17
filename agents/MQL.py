import numpy as np
from collections import defaultdict
import math
from humanoid_controller import *

class MQL():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate=0.01):
        self.action_size = action_size
        # Two separate Q-tables for topographic and social features
        self.q_values_topo = defaultdict(lambda: np.zeros(action_size))
        self.q_values_social = defaultdict(lambda: np.zeros(action_size))
        self.gamma = 0.99
        self.learning_rate = 0.2
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.controller = Controller()

    def sample_action(self, phi_state, eval=False):
        features_topo, features_social = phi_state
        key_topo = str(features_topo)
        key_social = str(features_social)
        q_topo = self.q_values_topo[key_topo]
        q_social = self.q_values_social[key_social]
        q_sum = q_topo + q_social

        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = int(np.argmax(q_sum))
        return action
    
    def phi(self, obs):
        goal = obs[0].flatten()
        agent = obs[1].flatten()
        walls = obs[2].flatten()
        goal[goal==1] = 1
        walls[walls==1] = 2
        features_topo = goal + walls
        features_social = agent
        return features_topo, features_social
    
    def update(
        self,
        phi_state,
        action: int,
        reward: float,
        terminated: bool,
        phi_next
    ):
        # Updates the Q-values for both topographic and social features.
        features_topo, features_social = phi_state
        next_topo, next_social = phi_next

        key_topo = str(features_topo)
        key_social = str(features_social)
        key_topo_next = str(next_topo)
        key_social_next = str(next_social)

        # Sum Q-values for next state for action selection
        q_topo_next = self.q_values_topo[key_topo_next]
        q_social_next = self.q_values_social[key_social_next]
        q_sum_next = q_topo_next + q_social_next

        future_q_value = (not terminated) * np.max(q_sum_next)
        q_topo = self.q_values_topo[key_topo][action]
        q_social = self.q_values_social[key_social][action]
        q_sum = q_topo + q_social

        temporal_difference = (
            reward + self.gamma * future_q_value - q_sum
        )

        # Update both Q-tables equally
        self.q_values_topo[key_topo][action] += self.learning_rate * temporal_difference / 2
        self.q_values_social[key_social][action] += self.learning_rate * temporal_difference / 2

    def act(self, env, obs, eval, reverse, upd_social=None):
        # Reset Humanoid Behaviour
        self.controller.reset()

        phi_state = self.phi(obs)
        total_reward = 0

        episode_length = 0
        collisions_humanoid = 0
        collisions_wall = 0

        while True:
            # Get Humanoid actions
            action_1, action_2, action_3, action_4 = self.controller.get_action(reverse)
            # Epsilon-greedy action selection
            action = self.sample_action(phi_state, eval)
            # Take action
            next_state, reward, done, truncated, info = env.step([action, action_1, action_2, action_3, action_4])

            # Update Humanoid actions
            self.controller.update(reward)

            phi_next =  self.phi(next_state[0])
            
            total_reward += reward[0]

            gnome = next((item for item in env.get_state()['Objects'] if item['Name'] == 'gnome'), None)
            collisions_humanoid = gnome['Variables']['collisions_humanoid']
            collisions_wall = gnome['Variables']['collisions_wall']
            
            self.update(phi_state, action, reward[0], done, phi_next)
            if done or truncated:
                break
            episode_length += 1
            phi_state = phi_next

        return  episode_length, total_reward, collisions_humanoid, collisions_wall
