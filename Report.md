# Collaboration and Competition using DDPG - Project Report

## The Environment

This project is built around the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent](https://github.com/jbdekker/ddpg-collaboration-and-competition/blob/81a9ed92b32ade847e89f75209bf6953bc04d188/models/test.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.  Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, your agents must get an average score of **+0.5** (over **100** consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least **+0.5**.

We've trained a DDPG agent to solve the double-agent (episodic) environment.

### State/observation space

The observation space consists of **8** variables corresponding to the position and velocity of the ball and racket. 

### Action space

Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

## Baseline - (random) agent

A random agent was provided to both test the environment and to set the baseline

```python
for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
```

As expected, the random agent's score averages around ``0``. To achive the **>=0.5** score, we need something better than a random walk.

## DDPG-agent implementation

The core of the code in this repository is based on the [ddpg-bipedal](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) and [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) agents. Some ammendments were made in order to make it work with the multi-agent Tennis environment.

Deep Deterministic Policy Gradient (DDPG) is a model-free off-policy algorithm for learning continous actions. It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network). It uses Experience Replay and slow-learning target networks from DQN, and it is based on DPG, which can operate over continuous action spaces. [[ref]](https://keras.io/examples/rl/ddpg_pendulum/)

The DRLND DDPG code was already written to utilize only a single agent, so the changes had to be made to make it work in a collaborative environment. Experience from the 2nd project helped a lot.

### Implementation details

The implementation / architecture of the DDPG agent is identical to the one developed for project #2:

1. Actor network - [[link]](https://github.com/jbdekker/ddpg-continuous-control/blob/27cb4e9595c02ad36538bb486cdbd831d7e3f4db/src/model.py#L14-L47)

    Multilayer (2-layer) perceptron model (128, 128) with batch normalisation, relu activation for the hidden layers and tanh activation on the output layer (output should be [-1, 1]).

    Experiments showed that a simple model structure worked best. Initially I tried more layers but this resulted in severely reduced training performance (and very unstable ...)

2. Critic network - [[link]](https://github.com/jbdekker/ddpg-continuous-control/blob/27cb4e9595c02ad36538bb486cdbd831d7e3f4db/src/model.py#L50)
    
    Multilayer (2-layer) perceptron model (128, 128) with batch normalisation, relu activation for the hidden layers and tanh activation on the output layer (output should be [-1, 1]).

    Again a limited number of layers to increase training performance & model stability
    
3. Gradient norm clipping [[link]](https://github.com/jbdekker/ddpg-continuous-control/blob/27cb4e9595c02ad36538bb486cdbd831d7e3f4db/src/ddpg_agent.py#L119)

    Clipping proved to be benificial to the training performance of the model

4. Hyperparameters

    A rough manual search of the hyperparameter space was performed. DDPG seemed especially sensitive to learning rate

    Final hyperparameters:

    ```python
        BUFFER_SIZE = int(1e6)  # replay buffer size
        BATCH_SIZE = 256        # minibatch size
        GAMMA = 0.99            # discount factor
        TAU = 1e-3              # for soft update of target parameters
        LR_ACTOR = 1e-4         # learning rate of the actor
        LR_CRITIC = 1e-4        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay
    ```

    Note: No extensive grid search was performed to optimize the hyperparameters. A naive manual search (small permutations) showed no significant improvement.

## Results

Final trained agent models:
-   [actor](models/checkpoint_actor.pth), [critic](models/checkpoint_critic.pth)
-   [scores np.save file](models/scores.npy)
-   [scores plot](models/scores.png)

The DDPG with experience replay (uniform sampling), batch normalisation & gradient clipping was able to consistently solve the environment in 1600-1800 episodes. The agent whose checkpoint is saved in ``models`` was able to solve the environment in **1720** episodes. A plot of the cumulative rewards (score) per training episode is shown below. It is clear that the training itself is very unstable & quite slow. The variance for the trained agent is very large, also the performance after **~1737** episodes drops down a bit and settles @ around **+0.25**. 

```bash
Environment solved in 1720 episodes! Average Score: 0.50
```

![Training progress](models/scores.png)

Below a video of the agent in action:

![Trained Agent](models/test.gif)

## Ideas for future work

1. **Tuning** The hyperparameter choice is somewhat arbitrary as no extensive grid-search was performed. 
2. **Reward tuning** The agents behave quite erratic when the ball is not near them. Tuning the reward structure a bit (e.g. penalty on excess movement) might smooth things down a bit.
3. **Prioritized experience replay** In uniform sampling (like what we used), all experiances have the same probability of being sampled. This  We can however assume that some experiences are more valuable than others (most experiences in this environment are not that probably not that important). With prioritized experience replay, the sampling distribution is proportional (or at least a function) of some measure of importance. 
4.  **Alternative model architectures** Other architectures can be tested (e.g. A3C, policy gradients) that might improve performance
