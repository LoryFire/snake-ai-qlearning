import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    # Rete semplice per stimare i valori Q
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Primo livello: prende lo stato e lo trasforma
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Secondo livello
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Passo 1: primo livello + ReLU
        x = F.relu(self.fc1(x))
        # Passo 2: secondo livello + ReLU
        x = F.relu(self.fc2(x))
        # Passo 3: output
        return self.out(x)
    