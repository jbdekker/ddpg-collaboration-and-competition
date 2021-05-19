# Collaboration and Competition using DDPG (Deep Deterministic Policy Gradient)

### Introduction

This project is built around the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent](https://github.com/jbdekker/ddpg-continuous-control/blob/1f4943e5f9e661ebf8a4c070262771ad538bce1f/models/test.gif)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of **8** variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

In this project we train a DDPG agent to solve the collaborative 2-agent (episodic) environment. This environment is considered solved when the agents achieve an average score of at least **+0.5** over **100** consecutive episodes (after taking the maximum over both agents).



### Getting Started

1. Clone this repository to a conveniant location. For the original Udacity repository, [click here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

3. Place the file the root of this repositoryand unzip (or decompress) the file.

4. Setup the python environment

    This project uses Python Poetry for dependency management. Poetry can be pip installed via:

    ```bash
    $ pip install --user poetry
    ```

    To setup the local virtual environment and install the dependencies from the ``poetry.lock`` file:

    ```bash
    $ poetry install
    ```

    In case you don't want to use Python Poetry, you're welcome to install the dependencies manually (see the ``pyproject.toml`` file).


## Training & testing

To train the DQN-agent, run (change the paths if neccesary): 

```bash
(venv) $ python train.py
```

A checkpoint of the actor & critic models is stored in the ``models`` directory, along with the training scores per episode (``numpy.save()`` object) and a plot of the training progress (episode vs score).

To visualize a run of the trained model in the Banana environment, run (change the paths if neccesary):

```bash
(venv) $ python test.py
```
