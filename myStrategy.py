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
gamma = 0.99
lmbda = 0.95
clip = 0.1
ent = 1e-3
epoch = 2
steps = 128
neps = 10000


class PPO(nn.Module):
    def __init__(self, env):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(5, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_pi = nn.Linear(hidden_size, 3)
        self.fc_v = nn.Linear(hidden_size, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self.device)

        self.env = env

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
        s_lst, a_lst, r_lst, s_next_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [
        ], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_next, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_next_lst.append(s_next)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_next, done_mask, prob_a = \
            torch.tensor(s_lst, dtype=torch.float32, device=self.device), \
            torch.tensor(a_lst, dtype=torch.int32, device=self.device), \
            torch.tensor(r_lst, dtype=torch.float32, device=self.device), \
            torch.tensor(s_next_lst, dtype=torch.float32, device=self.device), \
            torch.tensor(done_lst, dtype=torch.float32, device=self.device), \
            torch.tensor(prob_a_lst, dtype=torch.float32, device=self.device)
        self.data = []

        return s, a, r, s_next, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def update_net(self):
        s, a, r, s_next, done_mask, prob_a, (h_in1,
                                             h_in2), (h_out1, h_out2) = self.make_batch()
        h_in = (h_in1.detach(), h_in2.detach())
        h_out = (h_out1.detach(), h_out2.detach())

        for i in range(epoch):
            # advantage
            pi, v_s, _ = self.forward(s, h_in)

            with torch.no_grad():
                _, v_s_next, _ = self.forward(s_next, h_out)
                td_target = r + gamma * v_s_next.squeeze(1) * done_mask
                delta = td_target - v_s.squeeze(1)

                advantage_lst = []
                advantage = 0.0
                for t in range(len(delta) - 1, -1, -1):
                    advantage = gamma * lmbda * advantage + delta[t]
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                advantage = torch.tensor(
                    advantage_lst, dtype=torch.float32, device=self.device)

                returns = v_s.flatten() + advantage

            # loss
            dist = Categorical(pi.squeeze(1))
            log_pi_a = dist.log_prob(a.squeeze(1))
            # a/b == exp(log(a)-log(b))
            ratio = torch.exp(log_pi_a - torch.log(prob_a.squeeze(1)))
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantage
            
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s.flatten(), returns) + ent * -dist.entropy()

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

    def train(self):
        for n_epi in range(neps):
            score = 0
            
            h_out = (torch.zeros([1, 1, hidden_size], dtype=torch.float32, device=self.device),
                     torch.zeros([1, 1, hidden_size],  dtype=torch.float32, device=self.device))
            s = self.env.reset()
            done = False
            
            while not done:
                for t in range(steps):
                    h_in = h_out
                    prob, v, h_out = self.forward(torch.from_numpy(s).to(
                        dtype=torch.float32, device=self.device), h_in)
                    prob = prob.view(-1)
                    
                    m = Categorical(prob)
                    a = m.sample().item() 
                    s_next, r, done, info = env.step(a - 1) # -1, 0, 1

                    self.put_data(
                        (s, a, r, s_next, prob[a].item(), h_in, h_out, done))
                    s = s_next

                    score += r

                    if done:
                        break

                self.update_net()

            print("# of episode :{}, score : {:.1f}".format(
                n_epi, score))


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
        reward = self.hoding * (next_price - cur_price) - transFee
        return reward

    def _observation(self, cur_index):
        if self.log_diff:
            observation = self.logDailyOhlcv.loc[cur_index,
                                                 feature_list].values
        else:
            observation = self.dailyOhlcv.loc[cur_index,
                                              feature_list].values
        return observation.astype(np.float32)


def myStrategy(dailyOhlcvFile, minutelyOhlcvFile, openPrice):
    windowsSize = 180
    pastData = dailyOhlcvFile.loc[-windowsSize:, feature_list].values

    return 1


if __name__ == "__main__":
    dailyOhlcvFile = sys.argv[1]
    env = TradingEnv(dailyOhlcvFile, log_diff=False)
    agent = PPO(env)
    agent.train()