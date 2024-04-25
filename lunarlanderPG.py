"""
Author: malonelin
Date:   2024.04.18
"""

import gymnasium as gym
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from loguru import logger as log
from tensorboardX import SummaryWriter as SW

GAMMA = 0.988  # discount factor
LR = 0.013  # learning rate
LR_DECAY_EP2GAMMA = {
    400 : 1,
    750 : 1,
    1100: 1 }
LR_DECAY_AVGREWARD2GAMMA = {
    100 : 1,
    150 : 1,
    200 : 1 }

# ENV_NAME = 'CartPole-v1'
ENV_NAME = 'LunarLander-v2'
EPISODE = 100000        # max episode
TIMEOUT_STEP = 390      # timeout max step
STEP = TIMEOUT_STEP     # max step in an episode
HUMAN_TEST_EP = 0               # max human test episode
HUMAN_TEST_STEP = TIMEOUT_STEP  # max human test steps in an episode
AVG_STAT_EP = 10        # avg stat episode

TEST_MODE = True
TEST_MODE_W_FILE_NAME = "w/w_best.pt_rw270_lr0.013000_gm0.9880_ep3000"
# TEST_MODE_RENDER = 'human'
TEST_MODE_RENDER = None
TEST_MODE_EP = 200               # max episode in TEST_MODE
TEST_MODE_STEP = TIMEOUT_STEP   # max test steps in an episode in TEST_MODE

log.add('log/info_{time}.log')
log.info(f'hyparam: GAMMA{GAMMA:>4}. LR:{LR:.6f}. STEP:{STEP}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f'using:{device}')

W_FINENAME = 'w/w.pt'
BEST_W_FINENAME = 'w/w_best.pt'
sw = SW('tensorboardX')

class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        hid_size = action_dim * state_dim
        self.fc1 = nn.Linear(state_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)

class PGAgent(object):
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        self.max_avg_reward = 150
        self.learn_cnt = -1
        self.test_cnt = 0

    def add_test_cnt(self):
        self.test_cnt += 1

    def get_test_cnt(self):
        return self.test_cnt
    
    def get_learn_cnt(self):
        return self.learn_cnt

    def save_w(self):
        torch.save(self.network.state_dict(), W_FINENAME)

    def save_best_w(self, avg_reward, episode):
        if avg_reward > self.max_avg_reward:
            file_name = format(f'{BEST_W_FINENAME}_rw{avg_reward:.0f}_lr{self.get_lr():.6f}_gm{GAMMA:.4f}_ep{episode}')
            torch.save(self.network.state_dict(), file_name)
            log.success(f'save best file: {file_name}')
            self.max_avg_reward = avg_reward

    def load_w(self, w_f_n = None):
        w_file_name = W_FINENAME
        if w_f_n != None:
            w_file_name = w_f_n
        if not os.path.exists(w_file_name):
            log.error(f'no w file: {w_file_name}')
        self.network.load_state_dict(torch.load(w_file_name))
        log.info(f'load w file: {w_file_name}')

    def check_ep_reduce_lr(self, episode):
        for ep, gamma in LR_DECAY_EP2GAMMA.items():
            if ep == episode and 1 != gamma:
                self.reduce_lr(reduce_rate = gamma)
                break

    def check_avg_reward_reduce_lr(self, avg_reward):
        for avg_r, gamma in LR_DECAY_AVGREWARD2GAMMA.items():
            if avg_reward > avg_r and 1 != gamma:
                self.reduce_lr(reduce_rate = gamma)
                LR_DECAY_AVGREWARD2GAMMA.pop(avg_r)
                break

    def reduce_lr(self, reduce_rate = 0.1):
        old_lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] *= reduce_rate
        log.info(f'reduce LR from {old_lr:.6f} to: {self.get_lr():.6f}')

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        act_net = self.network.forward(observation)
        with torch.no_grad():
            act_p = F.softmax(act_net, dim=0).cuda().data.cpu().numpy()
        action = np.random.choice(range(act_p.shape[0]), p=act_p)
        return action

    def add_step_data(self, s, a, r, is_timeout_step):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
        if is_timeout_step:
            self.ep_rs[-1] = -100.0

    def normalize(self, t):
        t.sub_(torch.mean(t)).div_(torch.std(t))

    def learn(self):
        q_reward = torch.FloatTensor(np.zeros_like(self.ep_rs))
        tmp_reward = 0
        for t in reversed(range(0, len(self.ep_rs))):
            tmp_reward = tmp_reward * GAMMA + self.ep_rs[t]
            q_reward[t] = tmp_reward
        q_reward = q_reward.to(device)
        self.normalize(q_reward)

        obs = torch.FloatTensor(np.array(self.ep_obs)).to(device)
        acts = torch.LongTensor(self.ep_as).to(device)
        act_net = self.network.forward(obs)
        act_p = F.cross_entropy(input=act_net, target=acts, reduction='none')

        loss = torch.mean(act_p * q_reward)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.learn_cnt += 1

        sw.add_scalar('loss', loss.item(), self.get_learn_cnt())

class Stat():
    def __init__(self):
        # stat every AVG_STAT_EP
        self.avg_reward = 0
        self.avg_steps = 0
        self.gt200_cnt = 0
        self.truncated_cnt = 0

        # stat every ep
        self.land_reward = 0
        self.land_steps = 0

    def clear_all(self):
        self.__init__()

    def clear_ep_stat(self):
        self.land_reward = 0
        self.land_steps = 0

    def post_handle(self, ep):
        agent.check_ep_reduce_lr(ep)

        sw.add_scalar('land_reward', stat.land_reward, ep)
        sw.add_scalar('step', stat.land_steps, ep)
        sw.add_scalar('lr', agent.get_lr(), ep)
        self.clear_ep_stat()
        if ep % AVG_STAT_EP == 0 and ep != 0:
            self.avg_reward /= AVG_STAT_EP
            self.avg_steps /= AVG_STAT_EP
            agent.check_avg_reward_reduce_lr(self.avg_reward)
            agent.save_w()
            agent.save_best_w(self.avg_reward, ep)

            sw.add_scalar('avg_reward', self.avg_reward, ep)
            log.info(f'ep:{ep:>4}. lc:{agent.get_learn_cnt():>4}. lr:{agent.get_lr():.6f}. Test reward>=200:({self.gt200_cnt:>2}/{AVG_STAT_EP}). Avg steps(maxcnt):{self.avg_steps:>3.0f}({self.truncated_cnt:>2}/{AVG_STAT_EP}). Avg reward:{self.avg_reward:>7.2f}')
            self.clear_all()

if not TEST_MODE:
    env = gym.make(ENV_NAME, render_mode = None, max_episode_steps = TIMEOUT_STEP)
    agent = PGAgent(env)
    env_t = gym.make(ENV_NAME, render_mode = TEST_MODE_RENDER, max_episode_steps = TIMEOUT_STEP)
    stat = Stat()

def train():
    for episode in range(EPISODE):        
        state , _ = env.reset()
        steps = 0
        for steps in range(STEP):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.add_step_data(state, action, reward, truncated)
            state = next_state

            stat.avg_reward += reward
            stat.land_reward += reward
            if done or truncated:
                if truncated:
                    stat.truncated_cnt += 1
                if stat.land_reward >= 200:
                    stat.gt200_cnt += 1
                break
        agent.learn()

        stat.land_steps = steps + 1
        stat.avg_steps += stat.land_steps
        stat.post_handle(episode)
        training_test(episode)

def training_test(episode):
    if (episode % 50 == 0 and episode != 0):
        for i in range(HUMAN_TEST_EP):
            state , _ = env_t.reset()
            for j in range(HUMAN_TEST_STEP):
                action = agent.get_action(state)
                state, reward, done, truncated, _ = env_t.step(action)
                if done or truncated:
                    break

def test_mode():
    env = gym.make(ENV_NAME, render_mode = TEST_MODE_RENDER, max_episode_steps = TIMEOUT_STEP)
    agent = PGAgent(env)
    agent.load_w(TEST_MODE_W_FILE_NAME)
    avg_reward = 0
    gt200_cnt = 0
    total_steps = 0
    ep = 0
    for ep in range(TEST_MODE_EP):
        land_reward = 0
        state , _ = env.reset()
        steps = 0
        for steps in range(TEST_MODE_STEP):
            action = agent.get_action(state)
            state, reward, done, truncated, _ = env.step(action)
            land_reward += reward
            if done or truncated:
                break
        avg_reward += land_reward
        total_steps += steps
        if land_reward >= 200:
            gt200_cnt += 1
            sw.add_scalars('test_reward_steps', {'reward': land_reward, 'land_steps': steps}, ep)
            sw.add_histogram('test_step2reward', land_reward, steps)
            sw.add_histogram('test_reward2step', steps, land_reward)
            sw.add_histogram('test_reward', land_reward, ep)
            sw.add_histogram('test_steps', steps, ep)
        log.info(f'ep {ep:>2}, steps: {steps:>2}, reward: {land_reward:>7.2f}')
    if ep > 0:
        test_cnt = ep + 1
        avg_reward /= test_cnt
        avg_steps = total_steps/test_cnt
        log.info(f'Test reward>=200:({gt200_cnt:>2}/{test_cnt}). Avg steps:{avg_steps:>3.0f}. Avg reward: {avg_reward:>7.2f}')

def main():
    if TEST_MODE:
        test_mode()
    else:
        train()

if __name__ == '__main__':
    main()
