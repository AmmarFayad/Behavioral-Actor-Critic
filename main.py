import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from BAC import BAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import pybullet as p2
p2.connect(p2.UDP)
import pybullet_envs
import torch.optim as optim
l=torch.nn.MSELoss()
ll=torch.nn.PairwiseDistance(p=2,keepdim=True)
from autoencoder import autoencoder

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetahBulletEnv-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.03, metavar='G', #0.05 hopper, 0.04 walker, 0.05 reacher, 0.26 Ant with normalization, 0.03 HalfCheetah
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=50, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1500000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', ####################4 for humanoid
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')  
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()
score=[]
sc=[]

# Environment
env = gym.make(args.env_name)
#seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)

# Agent
agent = BAC(env.observation_space.shape[0], env.action_space, args)
enco=autoencoder(env.observation_space.shape[0]+env.action_space.shape[0])
c=optim.Adam(enco.parameters(), lr=0.003, weight_decay=0.001) 
#TensorboardX
writer = SummaryWriter(logdir='runs/{}_BAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
x=0
for i_episode in itertools.count(1):
    
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # SamNo documeple random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                if total_numsteps>30000:
                    
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parametersafter(memory, args.batch_size, updates,env,enco)
                    
                        
                else:
                    
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parametersbefore(memory, args.batch_size, updates)  
                    
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action.flatten()) # Step
        episode_steps += 1
        total_numsteps += 1
        x+=1
        episode_reward += reward

        
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    
    
    #Update the autoencoder network Periodically
    if i_episode% 5 ==0 and total_numsteps>args.start_steps :
        episodes=15
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action.flatten())
                a=torch.Tensor(action.flatten()).unsqueeze(0)
                cat=torch.cat((torch.Tensor(state).unsqueeze(0),a),dim=-1)
                s=enco(cat)
                f=l(cat,s)
                
                c.zero_grad()
                f.backward(retain_graph=True)
                c.step()
                state = next_state
        
    
    
agent.save_model(args.env_name,enco,suffix="")
env.close()

