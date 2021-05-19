import torch
import numpy as np

from unityagents import UnityEnvironment

from src.ddpg_agent import Agent


env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
postfix = ""

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = env_info.vector_observations.shape[1]
num_agents = len(env_info.agents)

agent = Agent(
    state_size=state_size,
    action_size=action_size,
    random_seed=42,
)

agent.actor_local.load_state_dict(torch.load(f"models/checkpoint_actor{postfix}.pth"))
agent.critic_local.load_state_dict(torch.load(f"models/checkpoint_critic{postfix}.pth"))

scores = np.zeros(num_agents)
while True:
    action = agent.act(states)
    env_info = env.step(action)[brain_name]
    states = env_info.vector_observations
    dones = env_info.local_done
    scores += env_info.rewards

    if any(dones):
        break

print(f"Max agent score (over all agents) this episode: {np.max(scores)}")

env.close()
