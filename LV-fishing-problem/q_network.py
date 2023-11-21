import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Initialize fully connected layers for the Q-network
        self.fc1 = nn.Linear(state_size + 1, 32) # +1 for subsystem index
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state, subsystem_index, time_step):
        x = torch.cat([state, torch.tensor([subsystem_index])], dim=-1) # Concatenate state with subsystem index
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)