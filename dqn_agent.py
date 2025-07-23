import torch, random
import numpy as np
from collections import deque
from dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size):
        # Quanti numeri descrivono lo stato
        self.state_size = state_size
        # Quante azioni può fare il serpente
        self.action_size = action_size
        # Memoria per ricordare le esperienze
        self.memory = deque(maxlen=10000)
        # Parametri per l'apprendimento
        self.gamma = 0.9  # quanto conta il futuro
        self.epsilon = 1.0  # quanto esplora
        self.epsilon_min = 0.01  # minimo di esplorazione
        self.epsilon_decay = 0.995  # quanto diminuisce epsilon
        self.learning_rate = 0.001  # quanto impara la rete
        self.batch_size = 64  # quante esperienze usa ogni volta

        # Sceglie GPU se c'è, altrimenti CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Crea la rete neurale per stimare i Q-values
        self.model = DQN(state_size, 128, action_size).to(self.device)
        # Ottimizzatore per aggiornare la rete
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Funzione di errore per l'addestramento
        self.loss_fn = torch.nn.MSELoss()

    def act(self, state):
        # Sceglie un'azione: a volte a caso (esplora), a volte la migliore (sfrutta)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        s = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(s)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Salva l'esperienza per allenare la rete dopo
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Allena la rete usando esperienze a caso dalla memoria
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Calcola i Q attuali e quelli target
        q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0]
        q_target = rewards + self.gamma * next_q * (~dones)

        # Calcola l'errore e aggiorna la rete
        loss = self.loss_fn(q, q_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Riduce epsilon (meno esplorazione col tempo)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
