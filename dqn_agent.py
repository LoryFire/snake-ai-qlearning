import torch, random
import numpy as np
from collections import deque
from dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        # dimensioni stato/azione
        self.state_size = state_size
        self.action_size = action_size
        # memoria di esperienza per replay
        self.memory = deque(maxlen=10000)
        # parametri RL
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        # imposta device per PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # rete neurale per Q-learning
        self.model = DQN(state_size, 128, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        """Restituisce un'azione: epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(s)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Salva transizione nella memoria di esperienza."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Esegue l'allenamento su un batch casuale."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0]
        q_target = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(q, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # aggiorna epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
