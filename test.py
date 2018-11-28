import torch
import torch.optim as optim

import numpy as np

import os
from common.utils import load_model
from model import DQN

def test(env, args): 
    p1_current_model = DQN(env, args).to(args.device)
    p2_current_model = DQN(env, args).to(args.device)
    p1_current_model.eval()
    p2_current_model.eval()

    load_model(p1_current_model, args, 1)
    load_model(p2_current_model, args, 2)

    p1_reward_list = []
    p2_reward_list = []
    length_list = []

    for _ in range(30):
        (p1_state, p2_state) = env.reset()
        p1_episode_reward = 0
        p2_episode_reward = 0
        episode_length = 0
        while True:
            if args.render:
                env.render()
            from time import sleep
            sleep(0.2)

            p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), 0.0)
            p2_action = p2_current_model.act(torch.FloatTensor(p2_state).to(args.device), 0.0)

            actions = {"1": p1_action, "2": p2_action}

            (p1_next_state, p2_next_state), reward, done, _ = env.step(actions)

            (p1_state, p2_state) = (p1_next_state, p2_next_state)
            p1_episode_reward += reward[0]
            p2_episode_reward += reward[1]
            episode_length += 1

            if done:
                p1_reward_list.append(p1_episode_reward)
                p2_reward_list.append(p2_episode_reward)
                length_list.append(episode_length)
                break
    
    print("Test Result - p1/Reward {} p2/Reward Length {}".format(
        np.mean(p1_reward_list), np.mean(p2_reward_list)))
    