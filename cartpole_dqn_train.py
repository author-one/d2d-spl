import os
import torch
import torch.nn
import numpy as np
import random
import gym
import pickle
from collections import namedtuple
from collections import deque
from datetime import datetime
from typing import List, Tuple
from cartpole_utils import create_cartpole_env

NUM_EPISODES = 1000
min_epsilon = 0.01
batch_size = 64
gamma = 0.99
hidden_dim = 12
capacity = 50_000

class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.cursor] = Transition(state, action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class Agent(object):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        return torch.autograd.Variable(torch.Tensor(x))

    def select_action(self, states: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes loss and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss

def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.get_Q(next_states).data.numpy(), axis=1) * ~done
    Q_target = agent._to_variable(Q_target)
    return agent.train(Q_predict, Q_target)


def play_episode(env: gym.Env, agent: Agent, replay_memory: ReplayMemory, eps: float, batch_size: int) -> int:
    s = env.reset()
    done = False
    total_reward = 0

    while not done:
        a = agent.select_action(s, eps)
        s2, r, done, info = env.step(a)
        total_reward += r
        if done:
            r = -1
        replay_memory.push(s, a, r, s2, done)

        if len(replay_memory) > batch_size:
            minibatch = replay_memory.pop(batch_size)
            train_helper(agent, minibatch, gamma)
        s = s2
    return total_reward

def get_env_dim(env: gym.Env) -> Tuple[int, int]:
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    return input_dim, output_dim

def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

def save_model(path, dqn):
    file = open(path,'wb')
    pickle.dump(dqn, file)

def main():
    results_path = './pytorch_dqn_results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    print('start:', datetime.now().strftime('%d/%m/%y %H:%M:%S'))

    env = create_cartpole_env()
    input_dim, output_dim = get_env_dim(env)
    num_trials = 10
    for trial in range(num_trials):
        env.seed(trial)
        torch.manual_seed(trial)
        np.random.seed(trial)
        file = open(results_path + '/all-scores-' + str(trial).zfill(2) + '.txt','w')
        start_time = datetime.now()

        agent = Agent(input_dim, output_dim, hidden_dim)
        replay_memory = ReplayMemory(capacity)
        total = 0        
        for episode in range(1, NUM_EPISODES + 1):
            eps = epsilon_annealing(episode, NUM_EPISODES - 1, min_epsilon)
            ep_reward = play_episode(env, agent, replay_memory, eps, batch_size)
            total += ep_reward
            #print("[Episode: {:5}] Reward: {:5} ".format(episode, ep_reward))
            file.write(str(episode) + "," + str(ep_reward) + '\n')
        print('average reward:', total / NUM_EPISODES)
        
        end_time = datetime.now()
        delta = end_time - start_time
        print('Trial', trial, ', Pytorch DQN learning 1 took ' + str(delta.total_seconds()) + ' seconds')
        print('trial', trial, ', avg score:', total / (NUM_EPISODES))
        save_model(results_path + '/model' + str(trial).zfill(2) + '-' + str(NUM_EPISODES) + '.p', agent.dqn)

        ### 2nd batch
        start_time = datetime.now()
        for episode in range(1 + NUM_EPISODES, 2 * NUM_EPISODES + 1):
            eps = epsilon_annealing(episode, NUM_EPISODES - 1, min_epsilon)
            ep_reward = play_episode(env, agent, replay_memory, eps, batch_size)
            total += ep_reward
            #print("[Episode: {:5}] Reward: {:5} ".format(episode, ep_reward))
            file.write(str(episode) + "," + str(ep_reward) + '\n')
        file.close()
        print('average reward:', total / (2*NUM_EPISODES))
        
        end_time = datetime.now()
        delta = end_time - start_time
        print('Trial', trial, ', Pytorch DQN learning 2 took ' + str(delta.total_seconds()) + ' seconds')
        print('trial', trial, ', avg score:', total / (2 * NUM_EPISODES))
        save_model(results_path + '/model' + str(trial).zfill(2) + '-' + str(2*NUM_EPISODES) + '.p', agent.dqn)

if __name__ == '__main__':
    main()