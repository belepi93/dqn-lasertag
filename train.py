import torch
import torch.optim as optim
import torch.nn.functional as F

import time, os
import numpy as np
from collections import deque

from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model, save_model
from model import DQN
from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

def train(env, args, writer):
    p1_current_model = DQN(env, args).to(args.device)
    p1_target_model = DQN(env, args).to(args.device)
    update_target(p1_current_model, p1_target_model)
    p2_current_model = DQN(env, args).to(args.device)
    p2_target_model = DQN(env, args).to(args.device)
    update_target(p2_current_model, p2_target_model)

    if args.noisy:
        p1_current_model.update_noisy_modules()
        p1_target_model.update_noisy_modules()
        p2_current_model.update_noisy_modules()
        p2_target_model.update_noisy_modules()

    if args.load_model and os.path.isfile(args.load_model):
        load_model(p1_current_model, args, 1)
        load_model(p2_current_model, args, 2)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        p1_replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
        p2_replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        p1_replay_buffer = ReplayBuffer(args.buffer_size)
        p2_replay_buffer = ReplayBuffer(args.buffer_size)
    
    p1_state_deque = deque(maxlen=args.multi_step)
    p2_state_deque = deque(maxlen=args.multi_step)
    p1_reward_deque = deque(maxlen=args.multi_step)
    p1_action_deque = deque(maxlen=args.multi_step)
    p2_reward_deque = deque(maxlen=args.multi_step)
    p2_action_deque = deque(maxlen=args.multi_step)

    p1_optimizer = optim.Adam(p1_current_model.parameters(), lr=args.lr)
    p2_optimizer = optim.Adam(p2_current_model.parameters(), lr=args.lr)

    length_list = []
    p1_reward_list, p1_loss_list = [], []
    p2_reward_list, p2_loss_list = [], []
    p1_episode_reward, p2_episode_reward = 0, 0
    episode_length = 0

    prev_time = time.time()
    prev_frame = 1

    (p1_state, p2_state) = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        if args.noisy:
            p1_current_model.sample_noise()
            p1_target_model.sample_noise()
            p2_current_model.sample_noise()
            p2_target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        p1_action = p1_current_model.act(torch.FloatTensor(p1_state).to(args.device), epsilon)
        p2_action = p2_current_model.act(torch.FloatTensor(p2_state).to(args.device), epsilon)

        if args.render:
            env.render()

        actions = {"1": p1_action, "2": p2_action}
        (p1_next_state, p2_next_state), reward, done, _ = env.step(actions)


        p1_state_deque.append(p1_state)
        p2_state_deque.append(p2_state)
        if args.negative:
            p1_reward_deque.append(reward[0] - 1)
        else:
            p1_reward_deque.append(reward[0])
        p1_action_deque.append(p1_action)
        if args.negative:
            p2_reward_deque.append(reward[1] - 1)
        else:
            p2_reward_deque.append(reward[1])
        p2_action_deque.append(p2_action)

        if len(p1_state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(p1_reward_deque, args.gamma)
            n_state = p1_state_deque[0]
            n_action = p1_action_deque[0]
            p1_replay_buffer.push(n_state, n_action, n_reward, p1_next_state, np.float32(done))

            n_reward = multi_step_reward(p2_reward_deque, args.gamma)
            n_state = p2_state_deque[0]
            n_action = p2_action_deque[0]
            p2_replay_buffer.push(n_state, n_action, n_reward, p2_next_state, np.float32(done))

        (p1_state, p2_state) = (p1_next_state, p2_next_state)
        p1_episode_reward += (reward[0])
        p2_episode_reward += (reward[1])
        if args.negative:
            p1_episode_reward -= 1
            p2_episode_reward -= 1
        episode_length += 1

        if done or episode_length > args.max_episode_length:
            (p1_state, p2_state) = env.reset()
            p1_reward_list.append(p1_episode_reward)
            p2_reward_list.append(p2_episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("data/p1_episode_reward", p1_episode_reward, frame_idx)
            writer.add_scalar("data/p2_episode_reward", p2_episode_reward, frame_idx)
            writer.add_scalar("data/episode_length", episode_length, frame_idx)
            p1_episode_reward, p2_episode_reward, episode_length = 0, 0, 0
            p1_state_deque.clear()
            p2_state_deque.clear()
            p1_reward_deque.clear()
            p2_reward_deque.clear()
            p1_action_deque.clear()
            p2_action_deque.clear()

        if len(p1_replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(p1_current_model, p1_target_model, p1_replay_buffer, p1_optimizer, args, beta)
            p1_loss_list.append(loss.item())
            writer.add_scalar("data/p1_loss", loss.item(), frame_idx)

            loss = compute_td_loss(p2_current_model, p2_target_model, p2_replay_buffer, p2_optimizer, args, beta)
            p2_loss_list.append(loss.item())
            writer.add_scalar("data/p2_loss", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(p1_current_model, p1_target_model)
            update_target(p2_current_model, p2_target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, p1_reward_list, length_list, p1_loss_list)
            print_log(frame_idx, prev_frame, prev_time, p2_reward_list, length_list, p2_loss_list)
            p1_reward_list.clear(), p2_reward_list.clear(), length_list.clear()
            p1_loss_list.clear(), p2_loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()
            save_model(p1_current_model, args, 1)
            save_model(p2_current_model, args, 2)

    save_model(p1_current_model, args, 1)
    save_model(p2_current_model, args, 2)


def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    state = torch.FloatTensor(np.float32(state)).to(args.device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:
        q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            next_q_values = current_model(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-5
        loss = (loss * weights).mean()
    
    else:
        q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.01, 0.99)

        target_dist = projection_distribution(current_model, target_model, next_state, reward, done, 
                                              target_model.support, target_model.offset, args)

        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)

    target_next_q_dist = target_model(next_state)

    if args.double:
        next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]

    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)

    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret