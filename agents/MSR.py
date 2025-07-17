import numpy as np
from collections import defaultdict
import math
from humanoid_controller import *

class MSR():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate_topo=0.01, learning_rate_social=0.01, w_learning_rate_topo=0.01, w_learning_rate_social=0.01):
        self.action_size = action_size
        
        # Use dictionaries for dynamic state space
        # SR[state][action][next_state] = expected discounted visits to next_state
        self.SR_topo = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.SR_social = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Reward function R[state][action] = expected immediate reward
        self.R_topo = defaultdict(lambda: defaultdict(float))
        self.R_social = defaultdict(lambda: defaultdict(float))

        self.epsilon = epsilon
        self.gamma = gamma
        self.w_learning_rate_topo = w_learning_rate_topo
        self.learning_rate_topo = learning_rate_topo
        self.w_learning_rate_social = w_learning_rate_social
        self.learning_rate_social = learning_rate_social

        self.controller = Controller()

    def get_state_key(self, obs):
        # Convert observation to hashable state key
        phi_topo, phi_social = self.phi(obs)
        return tuple(phi_topo.astype(int)), tuple(phi_social.astype(int))   # Convert to int for discrete states

    def sample_action(self, state_key_topo, state_key_social, eval=False):
        # Sample action using epsilon-greedy with SR-based Q-values
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Compute Q-values: Q(s,a) = sum over s' of (SR_topo(s,a,s') * R_topo(s') + SR_social(s,a,s') * R_social(s'))
        Q_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            q_value = 0.0
            # Topographic SR
            for next_state in self.SR_topo[state_key_topo][a]:
                reward_next_state_topo = np.mean([self.R_topo[next_state][a_prime] for a_prime in self.R_topo[next_state]])
                q_value += self.SR_topo[state_key_topo][a][next_state] * reward_next_state_topo
            # Social SR
            for next_state in self.SR_social[state_key_social][a]:
                reward_next_state_social = np.mean([self.R_social[next_state][a_prime] for a_prime in self.R_social[next_state]])
                q_value += self.SR_social[state_key_social][a][next_state] * reward_next_state_social
            Q_values[a] = q_value

        return np.argmax(Q_values)

    def update_sr(self, state_key_topo, state_key_social, action, next_state_key_topo, next_state_key_social, done, upd_social):
        if not done:
            next_action = self.sample_action(next_state_key_topo, next_state_key_social)
            relevant_s_primes_topo = set(self.SR_topo[state_key_topo][action].keys()) | set(self.SR_topo[next_state_key_topo][next_action].keys()) | {next_state_key_topo}
            if upd_social:
                relevant_s_primes_social = set(self.SR_social[state_key_social][action].keys()) | set(self.SR_social[next_state_key_social][next_action].keys()) | {next_state_key_social}
        else:
            next_action = None
            relevant_s_primes_topo = set(self.SR_topo[state_key_topo][action].keys()) | {next_state_key_topo}
            if upd_social:
                relevant_s_primes_social = set(self.SR_social[state_key_social][action].keys()) | {next_state_key_social}

        # Update topo SR
        for s_prime in tuple(relevant_s_primes_topo):
            indicator = 1.0 if s_prime == next_state_key_topo else 0.0
            if not done and next_action is not None:
                future_sr = self.SR_topo[next_state_key_topo][next_action][s_prime]
            else:
                future_sr = 0.0
            target = indicator + self.gamma * future_sr
            current_sr = self.SR_topo[state_key_topo][action][s_prime]
            td_error = target - current_sr
            self.SR_topo[state_key_topo][action][s_prime] += self.learning_rate_topo * td_error

        # Update social SR
        if upd_social:
            for s_prime in tuple(relevant_s_primes_social):
                indicator = 1.0 if s_prime == next_state_key_social else 0.0
                if not done and next_action is not None:
                    future_sr = self.SR_social[next_state_key_social][next_action][s_prime]
                else:
                    future_sr = 0.0
                target = indicator + self.gamma * future_sr
                current_sr = self.SR_social[state_key_social][action][s_prime]
                td_error = target - current_sr
                self.SR_social[state_key_social][action][s_prime] += self.learning_rate_social * td_error

    def update_reward(self, state_key_topo, state_key_social, action, reward, upd_social):
        # Update both topographic and social reward functions
        # Topographic reward update
        current_r_topo = self.R_topo[state_key_topo][action]
        td_error_topo = reward - current_r_topo
        self.R_topo[state_key_topo][action] += self.w_learning_rate_topo * td_error_topo

        if upd_social:
            # Social reward update
            current_r_social = self.R_social[state_key_social][action]
            td_error_social = reward - current_r_social
            self.R_social[state_key_social][action] += self.w_learning_rate_social * td_error_social

    def phi(self, obs):
        # Extract features from observation for state representation
        goal = obs[0].flatten()
        agent = obs[1].flatten()
        walls = obs[2].flatten()
        goal[goal==1] = 1
        walls[walls==1] = 2
        features_topo = goal + walls
        features_social = agent
        return features_topo, features_social

    def act(self, env, obs, eval, reverse, upd_social=True):
        # Reset Humanoid Behaviour
        self.controller.reset()
        state_key_topo, state_key_social = self.get_state_key(obs)
        
        episode_reward = 0
        episode_length = 0
        collisions_humanoid = 0
        collisions_wall = 0

        while True:
            # Get Humanoid actions
            action_1, action_2, action_3, action_4 = self.controller.get_action(reverse)

            # Select action
            action = self.sample_action(state_key_topo, state_key_social, eval)
            
            # Take step
            next_state, reward, done, truncated, info = env.step([action, action_1, action_2, action_3, action_4])
            
            # Update Humanoid actions
            self.controller.update(reward)
            
            # Get next state
            next_state_key_topo, next_state_key_social = self.get_state_key(next_state[0])
            
            episode_reward += reward[0]
            
            # Get collision info
            robot = next((item for item in env.get_state()['Objects'] if item['Name'] == 'robot'), None)
            collisions_humanoid = robot['Variables']['collisions_humanoid']
            collisions_wall = robot['Variables']['collisions_wall']
            
            self.update_sr(
                state_key_topo, state_key_social, action,
                next_state_key_topo, next_state_key_social, done or truncated, upd_social
            )
            self.update_reward(next_state_key_topo, next_state_key_social, action, reward[0], upd_social)
            
            # Move to next state
            state_key_topo = next_state_key_topo
            state_key_social = next_state_key_social
            
            if done or truncated:
                break
            episode_length += 1
            
        return episode_length, episode_reward, collisions_humanoid, collisions_wall
