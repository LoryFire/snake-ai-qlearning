import matplotlib.pyplot as plt
from snakeRLfriendly import SnakeEnv
from dqn_agent import DQNAgent

env = SnakeEnv(render_mode=False) #render_mode=False grafica disabilitata
state_size = len(env.get_state())
action_size = 4
agent = DQNAgent(state_size, action_size)

episodes = 500
scores = []

for episode in range(episodes):
    env.reset()
    state = env.get_state()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        reward = env.step(action)
        next_state = env.get_state()
        done = env.done

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward
        env.render()  # mostra in tempo reale

    scores.append(env.score)
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}, Score: {env.score}, Epsilon: {agent.epsilon:.3f}")

# Visualizza i punteggi
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN training progress')
plt.show()
