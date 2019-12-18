import sys
import random

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

hidden_size = 32
learning_rate = 1e-4
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 2
T_horizon = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(5, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, 3)
        self.fc_v = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, hidden_size)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc2(x)
        
        v = self.fc_v(x)
        pi = self.fc_pi(x)
        pi = F.softmax(pi, dim=2)
        return pi, v, lstm_hidden

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [
        ], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in,
                                              h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            # a/b == log(exp(a)-exp(b))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


feature_list = ["open", "high", "low", "close", "volume"]
transFee = 100
capital = 500000


class TradingEnv:
    def __init__(self, dailyOhlcvFile, log_diff=False):
        self.dailyOhlcv = pd.read_csv(dailyOhlcvFile)

        self.log_diff = log_diff
        if log_diff:
            self.logDailyOhlcv = self.dailyOhlcv.copy(deep=True)
            for name in feature_list:
                self.logDailyOhlcv[name] = np.log(
                    self.dailyOhlcv[name]) - np.log(self.dailyOhlcv[name].shift(1))

        self.reset()

    def reset(self):
        self.capital = capital
        self.hoding = 0
        self.cur_index = 0

        return self._observation(self.cur_index)

    def step(self, action):
        self.cur_index += 1
        done = self.cur_index >= len(self.dailyOhlcv)

        if not done:
            observation = self._observation(self.cur_index)

            cur_price, next_price = self.dailyOhlcv.loc[self.cur_index-1:self.cur_index,
                                                        ["open"]].values.ravel().tolist()
            if action == 1 and self.capital > transFee:
                self.hoding += (self.capital - transFee) / cur_price
                self.capital = 0
                reward = self._reward(cur_price, next_price)
            elif action == -1 and self.hoding * cur_price > transFee:
                self.capital += self.hoding * cur_price - transFee
                self.hoding = 0
                reward = self._reward(cur_price, next_price)
            else:
                reward = 0
        else:
            observation = self.reset()
            reward = 0

        info = {'capital': self.capital, 'holding': self.hoding}
        return observation, reward, done, info

    def _reward(self, cur_price, next_price):
        reward =  self.hoding * (next_price - cur_price) - transFee
        return reward

    def _observation(self, cur_index):
        if self.log_diff:
            observation = self.logDailyOhlcv.loc[cur_index,
                                                 feature_list].values
        else:
            observation = self.dailyOhlcv.loc[cur_index,
                                              feature_list].values
        return observation


def myStrategy(dailyOhlcvFile, minutelyOhlcvFile, openPrice):
    windowsSize = 180
    pastData = dailyOhlcvFile.loc[-windowsSize:, feature_list].values

    return 1


if __name__ == "__main__":
    dailyOhlcvFile = sys.argv[1]
    env = TradingEnv(dailyOhlcvFile, log_diff=False)

    done = False
    state = env.reset()
    print(state)

    while not done:

        action = random.choice([-1, 0, 1])
        state, reward, done, info = env.step(action)

        print(action, reward, info)
        print(state)
