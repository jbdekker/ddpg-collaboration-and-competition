import time
import torch
import numpy as np

from collections import deque
from unityagents import UnityEnvironment

from src.ddpg_agent import Agent

import matplotlib.pyplot as plt


env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
postfix = ""

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)

agent = Agent(
    state_size=state_size,
    action_size=action_size,
    random_seed=42,
)


def ddpg(n_episodes=3000):
    scores_deque = deque(maxlen=100)
    total_scores = []

    best_average_score = 0
    improved = False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        actions = agent.act(states)

        average_score = 0
        scores = np.zeros(num_agents)
        improved = False

        agent.reset()
        while True:
            next_actions = agent.act(states)

            env_info = env.step(next_actions)[brain_name]

            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done

            # add a small penalty for moving to promote smooth movements
            deltas = [np.abs(np.mean(a - b)) for a, b in zip(next_actions, actions)]

            rewards = rewards - 0.001 * deltas

            for state, action, reward, next_state, done in zip(
                states, next_actions, rewards, next_states, dones
            ):
                agent.step(state, action, reward, next_state, done)

            states = next_states
            actions = next_actions
            scores += rewards

            if np.any(dones):
                break

        scores_deque.append(scores)
        total_scores.append(scores)

        # we take the maximum average score over both agents
        average_score = np.max(np.mean(scores_deque, axis=0))

        # keep track of the best score
        if average_score > best_average_score:
            best_average_score = average_score
            improved = True

        # save the model if it improves its top score
        if average_score > 0.5 and improved:
            torch.save(
                agent.actor_local.state_dict(),
                f"models/checkpoint_actor{postfix}.pth",
            )
            torch.save(
                agent.critic_local.state_dict(),
                f"models/checkpoint_critic{postfix}.pth",
            )
            print(
                f"\n\n Environment solved in {i_episode} episodes! "
                f"Average Score: {average_score:.2f}"
            )

        print(
            f"\rEpisode {i_episode}"
            f"\tAverage Score: {average_score:.2f}"
            f"\tScore: {np.max(scores):.2f}"
            f"\tBest average score: {best_average_score:.2f}",
            end="",
        )

    return total_scores


scores = ddpg()
scores = np.array(scores)

np.save(f"models/scores{postfix}", scores)


def moving_average(x, w):
    y = np.convolve(x, np.ones(w), "valid") / w
    y = np.pad(y, (len(x) - len(y), 0), "constant", constant_values=np.nan)
    return y


fig, ax = plt.subplots(1, 1)

x = np.arange(len(scores))
for agent in range(scores.shape[1]):
    ax.plot(x, scores[:, agent], label=f"Agent #{agent}")
    ax.plot(
        x,
        moving_average(scores[:, agent], 100),
        label=f"Agent #{agent} - 100 episode average",
    )

ax.legend(loc="upper left")
ax.set_ylabel("Score")
ax.set_xlabel("Episode #")

plt.savefig(f"models/scores{postfix}.png")
plt.show()

env.close()
