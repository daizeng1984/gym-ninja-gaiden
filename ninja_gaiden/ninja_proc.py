import numpy as np
import cv2
from collections import deque
from torch.multiprocessing import Process
from ninja_gaiden.ninja_env import DEFAULT_MOVEMENT


class NesGymProc(Process):
    def __init__(
            self,
            env,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84):
        super(NesGymProc, self).__init__()
        self.daemon = True
        self.env = env
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(NesGymProc, self).run()
        while True:
            action = self.child_conn.recv()
            if self.is_render:
                self.env.render()
            obs, reward, done, info = self.env.step(action)

            # reward range -15 ~ 15
            log_reward = reward / 15
            self.rall += log_reward

            r = 0.

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(obs)

            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print(
                    "[Episode {}({})] Step: {}  Reward: {}  \
                    Recent Reward: {}  Stage: {} current x:{} \
                    max x:{}"
                    .format(
                        self.episode,
                        self.env_idx,
                        self.steps,
                        self.rall,
                        np.mean(
                            self.recent_rlist),
                        info['stage'],
                        info['screen_x'],
                        self.max_pos))

                self.history = self.reset()
            self.child_conn.send(
                [self.history[:, :, :], r, False, done, log_reward])

    def reset(self):
        self.steps = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        self.get_init_state(self.env.reset())
        return self.history[:, :, :]

    def pre_proc(self, X):
        # grayscaling
        x = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
        # resize
        x = cv2.resize(x, (self.h, self.w))
        x = np.float32(x) * (1.0 / 255.0)

        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)
