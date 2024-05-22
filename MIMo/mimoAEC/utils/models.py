import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import random
from collections import namedtuple, deque


def init_weights(layer):
    '''
    Weight initialization for networks.
    '''
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
    elif isinstance(layer, nn.ConvTranspose2d):
        nn.init.xavier_normal_(layer.weight)

def sparse_loss(encoding):
    '''
    Compute sparse loss using L1-norm
    '''
    loss = torch.mean(torch.abs(encoding))
    return loss

# --- Autoencoder ---

class Autoencoder(nn.Module):

    def __init__(self, input_dim=2, cnn1_dim=96, cnn2_dim=24):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_dim, cnn1_dim, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn1_dim, cnn2_dim, kernel_size=(1,1)),
            nn.ReLU(),
        )
        self.encoder_cnn.apply(init_weights)

        ### Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(cnn2_dim, cnn1_dim, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(cnn1_dim, input_dim, kernel_size=(4,4), stride=2),
            nn.Tanh()
        )
        self.decoder_cnn.apply(init_weights)

    def forward(self, x):
        enc = self.encoder_cnn(x)
        dec = self.decoder_cnn(enc)
        return enc, dec

class MagnoAutoencoder(nn.Module):

    def __init__(self, input_dim=4, cnn1_dim=192, cnn2_dim=48):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_dim, cnn1_dim, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            nn.Conv2d(cnn1_dim, cnn2_dim, kernel_size=(1,1)),
            nn.ReLU(),
        )
        self.encoder_cnn.apply(init_weights)

        ### Decoder
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(cnn2_dim, cnn1_dim, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(cnn1_dim, input_dim, kernel_size=(4,4), stride=2),
            nn.Tanh()
        )
        self.decoder_cnn.apply(init_weights)

    def forward(self, x):
        enc = self.encoder_cnn(x)
        dec = self.decoder_cnn(enc)
        return enc, dec

# --- Actor Critic ---

class Critic(nn.Module):
    def __init__(self, learning_rate, n_actions=1, input_dim=24, cnn_dim=24):
        super().__init__()
        self.n_actions = n_actions

        self. convpool = nn.Sequential(
            nn.Conv2d(input_dim, cnn_dim, kernel_size=(2,2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.convpool.apply(init_weights)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_1 = nn.Sequential(
            nn.Linear(18*cnn_dim+n_actions, 256),
            nn.ReLU()
        )
        self.fc_1.apply(init_weights)
        self.fc_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc_2.apply(init_weights)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.critic_head.apply(init_weights)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observation_fine, observation_coarse, action):
        observation_fine = self.convpool(observation_fine)
        observation_coarse = self.convpool(observation_coarse)
        observation_fine = self.flatten(observation_fine)
        observation_coarse = self.flatten(observation_coarse)
        obs_action_value = torch.cat([observation_fine,observation_coarse,action], dim=1)
        obs_action_value = self.fc_1(obs_action_value)
        obs_action_value = self.fc_2(obs_action_value)
        critic = self.critic_head(obs_action_value)
        return critic

class Actor(nn.Module):
    def __init__(self, learning_rate, max_action=2, n_actions=1, input_dim=24, cnn_dim=24):
        super().__init__()
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self. convpool = nn.Sequential(
            nn.Conv2d(input_dim, cnn_dim, kernel_size=(2,2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.convpool.apply(init_weights)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_1 = nn.Sequential(
            nn.Linear(18*cnn_dim, 256),
            nn.ReLU()
        )
        self.fc_1.apply(init_weights)
        self.fc_2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc_2.apply(init_weights)
        self.mu = nn.Linear(256, self.n_actions)
        self.sigma = nn.Linear(256, self.n_actions)  

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  

    def forward(self, observation_fine, observation_coarse):
        observation_fine = self.convpool(observation_fine)
        observation_coarse = self.convpool(observation_coarse)
        observation_fine = self.flatten(observation_fine)
        observation_coarse = self.flatten(observation_coarse)
        observation = torch.cat([observation_fine,observation_coarse], dim=1)
        observation = self.fc_1(observation)
        observation = self.fc_2(observation)

        mu = self.mu(observation)
        sigma = self.sigma(observation)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample(self, observation_fine, observation_coarse, reparameterize=True, deterministic=True):
        '''
        Sample action using outputs from actor network
        '''
        mu, sigma = self.forward(observation_fine, observation_coarse)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        if deterministic:
            actions=mu
            
        action = torch.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        action = action * torch.tensor(self.max_action)

        return action, log_probs