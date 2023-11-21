import numpy as np
import random

from scipy.integrate import solve_ivp

class SwitchedSystemEnv:
    def __init__(self):
        # Parameter initialization
        self.state = np.array([0.5, 0.7, 0], dtype=np.float32) # Initial state
        self.subsystem_index = random.choice([0, 1]) # Initial subsystem mode
        self.time_step = 0.0
        self.time_limit = 12 # Time interval
        self.dt = 0.4 # Time step
        self.c0 = 0.4
        self.c1 = 0.2

        # Variables for recording the history of actions and states
        self.subsystem_index_history = []
        self.x0_history = []

    def dynamics(self, t, state, w):
        # System dynamic
        x0, x1, x2 = state
        dx0 = x0 - x0 * x1 - self.c0 * x0 * w
        dx1 = -x1 + x0 * x1 - self.c1 * x1 * w
        dx2 = (x0 - 1)**2 + (x1 - 1)**2
        return [dx0, dx1, dx2]

    def step(self, action):
        # Apply the chosen action and update the system state
        self.subsystem_index_history.append(self.subsystem_index)
        self.x0_history.append(self.state[0])
        self.final_subsystem = self.subsystem_index

        # ODE solver
        t_span = [self.time_step, self.time_step + self.dt]
        sol = solve_ivp(self.dynamics, t_span, self.state, args=(self.subsystem_index,), method='RK45')
        self.state = sol.y[:, -1]

        self.time_step += self.dt

        reward = -(self.state[0] - 1) ** 2 - (self.state[1] - 1) ** 2

        done = self.time_step >= self.time_limit
        
        self.subsystem_index = action # Update the subsystem index for next time step
        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment for a new episode
        self.state = np.array([0.5, 0.7, 0], dtype=np.float32)
        self.subsystem_index = random.choice([0, 1])
        self.time_step = 0.0

        self.subsystem_index_history.clear()
        self.x0_history.clear()
        return self.state
    
    def record_final_state_and_action(self):
        # Record the final state and action after the episode ends
        self.subsystem_index_history.append(self.subsystem_index_history[-1])
        self.x0_history.append(self.state[0])