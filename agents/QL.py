import numpy as np
from collections import defaultdict
import math
from humanoid_controller import *

class QL():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate=0.01):
        self.action_size = action_size
        self.q_values = defaultdict(lambda: np.zeros(action_size))
        self.gamma = 0.99
        self.learning_rate = 0.2
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.controller = Controller()

    def sample_action(self, state, eval=False):
        # Samples action using epsilon-greedy approach
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = int(np.argmax(self.q_values[state]))
        return action
    
    def phi(self, obs):
        goal = obs[0].flatten()
        agent = obs[1].flatten()
        walls = obs[2].flatten()
        goal[goal==1] = 2
        walls[walls==1] = 3
        features = agent + goal + walls
        return features
    
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        # Update the Q-value of an action
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )
        return temporal_difference

    def act(self, env, obs, eval, reverse, upd_social=None):
        # Reset Humanoid Behaviour
        self.controller.reset()

        phi_state = self.phi(obs)
        state = str(phi_state)
        total_reward = 0

        episode_length = 0
        collisions_humanoid = 0
        collisions_wall = 0

        while True:
            # Get Humanoid actions
            action_1, action_2, action_3, action_4 = self.controller.get_action(reverse)

            # Epsilon-greedy action selection
            action = self.sample_action(state, eval)
            # Take action
            next_state, reward, done, truncated, info = env.step([action, action_1, action_2, action_3, action_4])

            # Update Humanoid actions
            self.controller.update(reward)

            phi_next =  self.phi(next_state[0])
            next_state = str(phi_next)
            
            total_reward += reward[0]

            robot = next((item for item in env.get_state()['Objects'] if item['Name'] == 'robot'), None)
            collisions_humanoid = robot['Variables']['collisions_humanoid']
            collisions_wall = robot['Variables']['collisions_wall']
            
            self.update(state, action, reward[0], done, next_state)
            if done or truncated:
                break
            episode_length += 1
            state = next_state

        return  episode_length, total_reward, collisions_humanoid, collisions_wall