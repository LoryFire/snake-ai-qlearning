import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Rete feedforward semplice per approssimare Q(s, a)."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # livello d’ingresso
        x = F.relu(self.fc2(x))  # livello nascosto
        return self.out(x)       # output Q-values
    