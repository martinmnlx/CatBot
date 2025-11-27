import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################

    alpha = 0.4
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.9985
    min_epsilon = 0.01
    max_steps = 150
    training_duration_seconds = 20  # Time limit for the entire training process
    
    def choose_action(state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return env.action_space.sample()  # Explore: random action
        else:
            return np.argmax(q_table[state])  # Exploit: best action from Q-table
        
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    start_time = None
    if render == -1:
        start_time = time.time()
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
        state, _ = env.reset()
        done = False
        
        for step in range(max_steps):
            action = choose_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)

            # Manually compute reward
            if done and reward == 0:
                reward = 1.0  # Reached the goal
            else:
                reward = -0.01  # Small penalty for each step to encourage faster completion

            # Update Q-table using the Q-learning formula
            old_value = q_table[state][action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state][action] = new_value
            
            state = next_state
            
            if done:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Check if the training time limit has been reached
        if start_time is not None:
            if time.time() - start_time > training_duration_seconds:
                print(f"\nTraining time limit of {training_duration_seconds} seconds reached. Stopping at episode {ep}.")
                break
 
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table