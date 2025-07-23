import matplotlib.pyplot as plt
from snakeRLfriendly import SnakeEnv
from dqn_agent import DQNAgent

# Crea l'ambiente di gioco senza grafica
env = SnakeEnv(render_mode=False)
# Quanti numeri descrivono lo stato
state_size = len(env.get_state())
# Quante azioni può fare il serpente
action_size = 4
# Crea l'agente DQN
agent = DQNAgent(state_size, action_size)

episodes = 500  # Quante partite fa l'agente
scores = []  # Qui salvo i punteggi

for episode in range(episodes):
    env.reset()  # Riavvia la partita
    state = env.get_state()  # Prendi lo stato iniziale
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)  # Scegli un'azione
        reward = env.step(action)  # Fai l'azione e prendi la ricompensa
        next_state = env.get_state()  # Nuovo stato
        done = env.done  # La partita è finita?

        agent.remember(state, action, reward, next_state, done)  # Salva l'esperienza
        agent.replay()  # Allena la rete

        state = next_state  # Aggiorna lo stato
        total_reward += reward  # Somma le ricompense
        env.render()  # Mostra la partita (disattivato se render_mode=False)

    scores.append(env.score)  # Salva il punteggio
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.3f}")  # Stampa ogni 50 partite

# Grafico dei punteggi
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN training progress')
plt.show()
