import numpy as np
from snakeRLfriendly import SnakeEnv
import random

env = SnakeEnv()
# sinistra,destra,su,giù
n_actions = 4

q_table = {}

# iperparametri
episodes = 10000
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0  # esplorazione
epsilon_decay = 0.995
min_epsilon = 0.01

scores = []

for episode in range(episodes):
    env.reset()
    total_reward = 0
    done = False

    state = tuple(env.get_state())

    while not done:
        # Esplora o sfrutta?
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            # scegli l'azione migliore dalla tabella
            if state in q_table:
                action = np.argmax(q_table[state])
            else:
                action = random.randint(0, n_actions - 1)

        reward = env.step(action)
        new_state = tuple(env.get_state())
        done = env.done

        # se lo stato non esiste ancora nella tabella, inizializzalo
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)
        if new_state not in q_table:
            q_table[new_state] = np.zeros(n_actions)

        # aggiorna la Q-table
        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_state]) - q_table[state][action]
        )

        state = new_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"episodio {episode + 1}, punteggio: {env.score}, epsilon: {epsilon:.3f}")
    scores.append(env.score)

total_reward += reward

print("allenamento completato!")

#AGGIORNAMENTO VISUALIZZAZIONE DATI

import matplotlib.pyplot as plt

# Grafico punteggi
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(scores, label='Score per episodio')
plt.xlabel('Episodio')
plt.ylabel('Punteggio')
plt.title('Punteggio durante l\'allenamento')
plt.legend()
plt.show() 




