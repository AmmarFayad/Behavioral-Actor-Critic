# Behavioral Actor-Critic
We provide an implementation of the algorithm proposed in [arxiv link] using Pytorch. Behavioral Actor-Critic is a deep reinforcement learning framework for in continuous domains. The algorithm makes use of several concepts such as autoencoders (to formulate the intrinsic reward), and the off-policy method presented in [Soft Actor-Critic](https://arxiv.org/pdf/1801.01290.pdf) paper.

Our prominent contribution is presenting an adaptive temperature approach along with normalization of our data which, in turn, produced state-of-the-art results.
## Prerequisites
As mentioned in the paper, we measure the performance of our algorithm on a suite of PyBullet continuous control tasks, interfaced through OpenAI Gym.
