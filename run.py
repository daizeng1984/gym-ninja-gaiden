from ninja_gaiden import NesGymProc
from ninja_gaiden.ninja_env import _make_ninja_gaiden_gym
from dqn.agent import ActorAgent, RunningMeanStd, RewardForwardFilter, _make_train_data
from torch.multiprocessing import Pipe
from tensorboardX import SummaryWriter
import logging
import numpy as np
import torch.nn.functional as F
import torch
import time
import datetime
from collections import deque


if __name__ == '__main__':
    # Create dummpy env to see input size etc.
    env = _make_ninja_gaiden_gym()
    input_size = env.observation_space.shape
    output_size = env.action_space.n
    logging.info('input size: {}, output size: {}'
                 .format(input_size, output_size))
    env.close()

    # env_id = 'SuperMarioBros-v0'
    # movement = COMPLEX_MOVEMENT
    # env = BinarySpaceToDiscreteSpaceEnv(
    #     gym_super_mario_bros.make(env_id), movement)
    # input_size = env.observation_space.shape  # 4
    # output_size = env.action_space.n  # 2

    writer = SummaryWriter()
    use_cuda = True
    use_gae = True
    life_done = True

    is_load_model = True
    is_training = False

    is_render = True
    use_standardization = True
    use_noisy_net = False

    model_path = 'data/{}_{}.model.none'.format(
        'ninja-gaiden-v0',
        datetime.date.today().isoformat())
    load_model_path = 'data/ninja-gaiden-v0_2019-05-18.model.none'

    lam = 0.95
    num_worker = 1
    num_step = 128
    ppo_eps = 0.1
    epoch = 3
    batch_size = 256
    max_step = 1.15e8

    learning_rate = 0.001
    lr_schedule = False

    stable_eps = 1e-30
    entropy_coef = 0.02
    alpha = 0.99
    gamma = 0.99
    clip_grad_norm = 0.5

    # Curiosity param
    icm_scale = 10.0
    beta = 0.2
    eta = 1.0
    reward_scale = 1

    agent = ActorAgent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        learning_rate=learning_rate,
        epoch=epoch,
        use_cuda=use_cuda,
        use_noisy_net=use_noisy_net)
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(gamma)

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
        from gym.wrappers import Monitor
        env = Monitor(env, './video', force=True)
        work = NesGymProc(env, is_render, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    sample_episode = 0
    sample_rall = 0
    sample_i_rall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    while True:
        total_state, total_reward, total_done, \
            total_next_state, total_action = [], [], [], [], []
        global_step += (num_worker * num_step)

        for _ in range(num_step):
            if not is_training:
                time.sleep(0.05)

            agent.model.eval()
            agent.icm.eval()

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
            rewards = np.hstack(rewards) * reward_scale
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)

            # total reward = int reward + ext Resard
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
            sample_i_rall += intrinsic_reward[sample_env_idx]
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar(
                    'data/i-reward', sample_i_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_i_rall = 0
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

            # running mean int reward
            total_reward_per_env = np.array([discounted_reward.update(
                reward_per_step) for reward_per_step in total_reward.reshape([num_worker, -1]).T])
            total_reawrd_per_env = total_reward_per_env.reshape([-1])
            mean, std, count = np.mean(total_reward), np.std(
                total_reward), len(total_reward)
            reward_rms.update_from_moments(mean, std ** 2, count)

            # devided reward by running std
            total_reward /= np.sqrt(reward_rms.var)

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
                target, adv = _make_train_data(total_reward[idx * num_step:(idx + 1) * num_step],
                                              total_done[idx *
                                                         num_step:(idx + 1) * num_step],
                                              value[idx *
                                                    num_step:(idx + 1) * num_step],
                                              next_value[idx * num_step:(idx + 1) * num_step], use_gae)
                total_target.append(target)
                total_adv.append(adv)

            if use_standardization:
                adv = (adv - adv.mean()) / (adv.std() + stable_eps)

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
