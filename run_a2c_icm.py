import logging

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import datetime

from ninja_gaiden import NesGymProc
from ninja_gaiden.ninja_env import _make_ninja_gaiden_gym
from torch.multiprocessing import Pipe
from tensorboardX import SummaryWriter
from dqn.model import DeepCnnActorCriticNetwork, CnnActorCriticNetwork, CuriosityModel, Categorical

import torch.optim as optim

from collections import deque


class ActorAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=True):
        self.model = CnnActorCriticNetwork(
            input_size, output_size, use_noisy_net)
        if use_icm:
            self.icm = CuriosityModel(input_size, output_size)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        if use_icm:
            self.optimizer = optim.Adam(
                list(
                    self.model.parameters()) +
                list(
                    self.icm.parameters()),
                lr=learning_rate)
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.model.to(self.device)
        if use_icm:
            self.icm = self.icm.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(policy)

        return action

    def compute_intrinsic_reward(self, state, next_state, action):
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.output_size).to(
            self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
            [state, next_state, action_onehot])
        intrinsic_reward = eta * \
            ((real_next_state_feature - pred_next_state_feature).pow(2)).sum(1) / 2.
        return intrinsic_reward.data.cpu().numpy()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def forward_transition(self, state, next_state):
        state = torch.from_numpy(state).to(self.device)
        state = state.float()
        policy, value = self.model(state)

        next_state = torch.from_numpy(next_state).to(self.device)
        next_state = next_state.float()
        _, next_value = self.model(next_state)

        value = value.data.cpu().numpy().squeeze()
        next_value = next_value.data.cpu().numpy().squeeze()

        return value, next_value, policy

    def train_model(
            self,
            s_batch,
            next_s_batch,
            target_batch,
            y_batch,
            adv_batch):
        with torch.no_grad():
            s_batch = torch.FloatTensor(s_batch).to(self.device)
            next_s_batch = torch.FloatTensor(next_s_batch).to(self.device)
            target_batch = torch.FloatTensor(target_batch).to(self.device)
            y_batch = torch.LongTensor(y_batch).to(self.device)
            adv_batch = torch.FloatTensor(adv_batch).to(self.device)

        if use_standardization:
            adv_batch = (adv_batch - adv_batch.mean()) / \
                (adv_batch.std() + stable_eps)

        ce = nn.CrossEntropyLoss()
        # mse = nn.SmoothL1Loss()
        forward_mse = nn.MSELoss()
        # --------------------------------------------------------------------------------
        if use_icm:
            # for Curiosity-driven
            action_onehot = torch.FloatTensor(
                len(s_batch), self.output_size).to(
                self.device)
            action_onehot.zero_()
            action_onehot.scatter_(1, y_batch.view(len(y_batch), -1), 1)

            real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
                [s_batch, next_s_batch, action_onehot])

            inverse_loss = ce(pred_action, y_batch)
            forward_loss = forward_mse(
                real_next_state_feature,
                pred_next_state_feature)

        # --------------------------------------------------------------------------------
        # for multiply advantage
        policy, value = self.model(s_batch)
        m = Categorical(F.softmax(policy, dim=-1))

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * adv_batch

        # Entropy(for more exploration)
        entropy = m.entropy()

        # Critic loss
        mse = nn.MSELoss()
        critic_loss = mse(value.sum(1), target_batch)

        self.optimizer.zero_grad()

        # Total loss
        if use_icm:
            loss = lamb * (actor_loss.mean() + 0.5 * critic_loss) + \
                (1 - beta) * inverse_loss + beta * forward_loss
        else:
            loss = actor_loss.mean() + 0.5 * critic_loss - entropy_coef * entropy.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        self.optimizer.step()


def make_train_data(reward, done, value, next_value):
    discounted_return = np.empty([num_step])

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * \
                next_value[t] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

        # For Actor
        adv = discounted_return - value

    else:
        running_add = next_value[-1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t] + gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

        # For Actor
        adv = discounted_return - value

    return discounted_return, adv


if __name__ == '__main__':
    # Create dummpy env to see input size etc.
    env = _make_ninja_gaiden_gym()
    input_size = env.observation_space.shape
    output_size = env.action_space.n
    logging.info('input size: {}, output size: {}'
                 .format(input_size, output_size))
    env.close()


    writer = SummaryWriter()
    use_cuda = False
    use_gae = True
    life_done = True

    is_load_model = False
    is_training = True

    is_render = True
    use_standardization = True
    use_noisy_net = True
    use_icm = False

    model_path = 'data/{}_{}-a2c.model'.format(
        'ninja-gaiden-v0',
        datetime.date.today().isoformat())
    load_model_path = 'data/ninja-gaiden-v0_2019-01-03-a2c.model'

    lam = 0.95
    num_worker = 8
    num_step = 5
    max_step = 1.15e8

    if use_icm:
        learning_rate = 0.001
    else:
        learning_rate = 0.00025
    lr_schedule = False

    stable_eps = 1e-30
    entropy_coef = 0.02
    alpha = 0.99
    gamma = 0.99
    clip_grad_norm = 0.5

    # Curiosity param
    lamb = 0.1
    beta = 0.2
    eta = 0.01

    agent = ActorAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net)

    if is_load_model:
        if use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'))

    if not is_training:
        agent.model.eval()

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        env = _make_ninja_gaiden_gym()
        work = NesGymProc(env, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    while True:
        total_state, total_reward, total_done, total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            if not is_training:
                time.sleep(0.05)
            actions = agent.get_action(states)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards = [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd)
                log_rewards.append(lr)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            if use_icm:
                intrinsic_reward = agent.compute_intrinsic_reward(
                    states, next_states, actions)
                rewards += intrinsic_reward

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0

        if is_training:
            total_state = np.stack(total_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.stack(total_next_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_reward = np.stack(total_reward).transpose().reshape([-1])
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_done = np.stack(total_done).transpose().reshape([-1])

            value, next_value, policy = agent.forward_transition(
                total_state, total_next_state)

            # logging utput to see how convergent it is.
            policy = policy.detach()
            m = F.softmax(policy, dim=-1)
            recent_prob.append(m.max(1)[0].mean().cpu().numpy())
            writer.add_scalar(
                'data/max_prob',
                np.mean(recent_prob),
                sample_episode)

            total_target = []
            total_adv = []
            for idx in range(num_worker):
                target, adv = make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                              total_done[idx *
                                                         num_step:(idx + 1) * num_step],
                                              value[idx *
                                                    num_step:(idx + 1) * num_step],
                                              next_value[idx * num_step:(idx + 1) * num_step])
                total_target.append(target)
                total_adv.append(adv)

            agent.train_model(
                total_state,
                total_next_state,
                np.hstack(total_target),
                total_action,
                np.hstack(total_adv))

            # adjust learning rate
            if lr_schedule:
                new_learing_rate = learning_rate - \
                    (global_step / max_step) * learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_learing_rate
                    writer.add_scalar(
                        'data/lr', new_learing_rate, sample_episode)

            if global_step % (num_worker * num_step * 100) == 0:
                torch.save(agent.model.state_dict(), model_path)
