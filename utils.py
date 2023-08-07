import math

import gymnasium as gym

import numpy as np
import torch
from torch import nn


class BetaHead(nn.Module):
    def __init__(self, in_features, action_size):
        super(BetaHead, self).__init__()
        
        self.fcc_c0 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c0.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c0.bias)
        
        self.fcc_c1 = nn.Linear(in_features, action_size)
        nn.init.orthogonal_(self.fcc_c1.weight, gain=0.01)
        nn.init.zeros_(self.fcc_c1.bias)
    
    def forward(self, x):
        c0 = torch.nn.functional.softplus(self.fcc_c0(x)) + 1.
        c1 = torch.nn.functional.softplus(self.fcc_c1(x)) + 1.
        return torch.distributions.Independent(
            torch.distributions.Beta(c1, c0), 1
        )


class GaussianHead(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.distributions.Normal(x, 0.1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        
        nn.init.orthogonal_(layer.weight.data, gain)
        
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0.0)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer.bias, "data"):
            torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def test_env(agent: torch.nn.Module, test_envs: gym.wrappers.RecordEpisodeStatistics, device, render=False):
    agent.eval()
    
    episode_reward = 0
    episode_length = 0
    episode_error = 0
    ave_reward = 0
    
    n_runs = len(test_envs.envs)
    
    not_done_flags = {i: True for i in range(n_runs)}
    
    intero_errors = np.zeros(n_runs)
    obs, info = test_envs.reset()
    obs = torch.Tensor(obs).to(device)
    
    while np.any(list(not_done_flags.values())):
        
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs)
        
        obs, reward, done, truncated, info = test_envs.step(action.cpu().numpy())
        done = done | truncated
        
        if render:
            test_envs.envs[0].render()
        
        obs = torch.Tensor(obs).to(device)
        
        if "interoception" in info.keys():
            try:
                # TODO check this.
                intero_errors += np.stack(info["interoception"] ** 2).sum(axis=1)
            except TypeError:
                print(info["interoception"])
        
        if np.any(done):
            for i in np.where(info["_episode"])[0]:
                if not_done_flags[i] is True:
                    not_done_flags[i] = False
                    print(
                        f"TEST: episodic_return={info['episode']['r'][i]}, episodic_length={info['episode']['l'][i]}")
                    
                    episode_reward += info['episode']['r'][i]
                    episode_length += info['episode']['l'][i]
                    episode_error += intero_errors[i] / info['episode']['l'][i]
                    ave_reward += info['episode']['r'][i] / info['episode']['l'][i]
                    intero_errors[i] = 0
                
                if np.any(list(not_done_flags.values())) is False:
                    break
    
    episode_reward /= n_runs
    episode_length /= n_runs
    episode_error /= n_runs
    ave_reward /= n_runs
    
    agent.train()
    
    return episode_reward, episode_length, episode_error, ave_reward


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
