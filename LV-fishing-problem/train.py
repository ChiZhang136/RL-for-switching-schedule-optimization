import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from environment import SwitchedSystemEnv
from q_network import QNetwork


n_episode = 10000

def train(env, agent, episodes, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99):
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    epsilon = epsilon_start

    rewards = []
    losses = []
    
    for episode in range(episodes):
        total_reward = 0
        total_loss = 0
        step = 0
        done = False

        # Initialize the state
        full_state = torch.tensor(env.reset(), dtype=torch.float32)
        state = torch.tensor([full_state[0], full_state[1]], dtype=torch.float32)
        
        while not done:
            q_values = agent(state, env.subsystem_index, env.time_step) # Get Q-values from the current state

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                action = torch.argmax(q_values).item()

            # Take action and observe the next state and reward
            full_next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor([full_next_state[0], full_next_state[1]], dtype=torch.float32)

            # Calculate the target Q-value
            with torch.no_grad():
                q_values_next = agent(next_state, env.subsystem_index, env.time_step)
                q_max = torch.max(q_values_next).item()
                q_target = reward + gamma * q_max

            # Backpropagation
            optimizer.zero_grad()
            q_value = q_values[action]
            loss = F.mse_loss(q_value, torch.tensor(q_target, dtype=torch.float32))
            loss.backward()

            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            total_reward += reward
            step += 1
            state = next_state  

        epsilon = max(epsilon_end, epsilon_decay * epsilon) # Update epsilon for exploration-exploitation balance

        losses.append(total_loss / step)
        rewards.append(total_reward)
        env.record_final_state_and_action() # Record the final state and action
        
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}, Average Loss: {total_loss / step}, Total Reward: {total_reward}")

    # Plotting x1 values and subsystem indices
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    number_of_points = int(env.time_limit / env.dt) + 1
    time_points = np.linspace(0, env.time_limit, number_of_points)
    plt.step(time_points, env.subsystem_index_history[:len(time_points)], where='post', color='red', label='Discrete Action', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Discrete Action')
    plt.title('Discrete Actions over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(time_points, env.x0_history[:len(time_points)], color='green', label='x1 Value', linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('x1 Value')
    plt.title('x1 Values over Time')
    plt.legend()
    plt.grid(True)

    plt.show(block=False)

    # Plotting loss and reward values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

    plt.subplot(1, 2, 2)
    plt.plot(rewards, label='Reward')
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()

    plt.show(block=False)

    window_size = 100 # Size of the moving window for averaging rewards
    moving_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    #Plotting the moving average of rewards
    plt.figure()
    plt.plot(moving_avg_rewards)
    plt.title('Moving Average of Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')

    plt.show()


# Initialize environment and agent
env = SwitchedSystemEnv()
state_size = 2 # x_0 and x_1 are used as input of the network
action_size = 2
agent = QNetwork(state_size, action_size)

# Train the agent
train(env, agent, n_episode)