import numpy as np
from collections import defaultdict
import math
from humanoid_controller import *

class SR():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate=0.01, w_learning_rate=0.01):
        self.action_size = action_size
        
        # Use dictionaries for dynamic state space
        # SR[state][action][next_state] = expected discounted visits to next_state
        self.SR = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Reward function R[state][action] = expected immediate reward
        self.R = defaultdict(lambda: defaultdict(float))
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.w_learning_rate = w_learning_rate
        self.learning_rate = learning_rate

        self.controller = Controller()

    def get_state_key(self, obs):
        # Convert observation to hashable state key
        phi = self.phi(obs)
        return tuple(phi.astype(int))
    
    def sample_action(self, state_key, eval=False):
        # Sample action using epsilon-greedy with SR-based Q-values
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Compute Q-values: Q(s,a) = sum over s' of SR(s,a,s') * R(s')
        Q_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            q_value = 0.0
            for next_state in self.SR[state_key][a]:
                if next_state in self.R and len(self.R[next_state]) > 0:
                    reward_next_state = np.mean([self.R[next_state][a_prime] for a_prime in self.R[next_state]])
                else:
                    reward_next_state = 0.0
                q_value += self.SR[state_key][a][next_state] * reward_next_state
            Q_values[a] = q_value

        return np.argmax(Q_values)

    def update_sr(self, state_key, action, next_state_key, done):
        if not done:
            next_action = self.sample_action(next_state_key)
            relevant_s_primes = set(self.SR[state_key][action].keys()) | set(self.SR[next_state_key][next_action].keys()) | {next_state_key}
        else:
            next_action = None
            relevant_s_primes = set(self.SR[state_key][action].keys()) | {next_state_key}

        for s_prime in tuple(relevant_s_primes):
            indicator = 1.0 if s_prime == next_state_key else 0.0
            if not done and next_action is not None:
                future_sr = self.SR[next_state_key][next_action][s_prime]
            else:
                future_sr = 0.0
            target = indicator + self.gamma * future_sr
            current_sr = self.SR[state_key][action][s_prime]
            td_error = target - current_sr
            self.SR[state_key][action][s_prime] += self.learning_rate * td_error

    def update_reward(self, state_key, action, reward):
        # Update reward function
        current_r = self.R[state_key][action]
        td_error = reward - current_r
        self.R[state_key][action] += self.w_learning_rate * td_error

    def phi(self, obs):
        # Extract features from observation for state representation
        goal = obs[0].flatten()
        agent = obs[1].flatten()
        walls = obs[2].flatten()
        goal[goal==1] = 2
        walls[walls==1] = 3
        features = agent + goal + walls
        return features

    def act(self, env, obs, eval, reverse, upd_social=True):
        # Reset Humanoid Behaviour
        self.controller.reset()
        # Run episode with SR learning
        state_key = self.get_state_key(obs)
        
        episode_reward = 0
        episode_length = 0
        collisions_humanoid = 0
        collisions_wall = 0


        while True:
            # Get Humanoid actions
            action_1, action_2, action_3, action_4 = self.controller.get_action(reverse)

            
            # Select action
            action = self.sample_action(state_key, eval)
            
            # Take step
            next_state, reward, done, truncated, info = env.step([action, action_1, action_2, action_3, action_4])
            
            # Update Humanoid actions
            self.controller.update(reward)

            # Get next state
            next_state_key = self.get_state_key(next_state[0])
            
            episode_reward += reward[0]
            
            # Get collision info
            robot = next((item for item in env.get_state()['Objects'] if item['Name'] == 'robot'), None)
            collisions_humanoid = robot['Variables']['collisions_humanoid']
            collisions_wall = robot['Variables']['collisions_wall']
            
            #if not eval:
                # Update SR and reward function
            self.update_sr(state_key, action, next_state_key, done or truncated)
            self.update_reward(next_state_key, action, reward[0])
            
            # Move to next state
            state_key = next_state_key
            
            if done or truncated:
                break
            episode_length += 1
            
        return episode_length, episode_reward, collisions_humanoid, collisions_wall
