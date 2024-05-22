import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import random
from collections import namedtuple, deque
from PIL import Image
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader
from math import tan, atan, pi

from src.utils import get_device


def sparse_loss(encoding):
    '''
    Compute sparse loss using L1-norm
    '''
    loss = torch.mean(torch.abs(encoding))
    return loss


#####  AUTOENCODER  #####


class Encoder(nn.Module):
    
    def __init__(self, input_dim, cnn1_dim, cnn2_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(input_dim, cnn1_dim, kernel_size=(8,8), stride=4), 
            nn.ReLU(),
            nn.Conv2d(cnn1_dim, cnn2_dim, kernel_size=(1,1), stride=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, input_dim, cnn1_dim, cnn2_dim):
        super().__init__()

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(cnn2_dim, cnn1_dim, kernel_size=(3,3), stride=2),    # C*15*15
            nn.ReLU(),
            nn.ConvTranspose2d(cnn1_dim, input_dim, kernel_size=(4,4), stride=2),   # C*32*32
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decoder_cnn(x)
        return x


class Autoencoder():

    def __init__(self, device, 
                 learning_rate, batch_size, minibatch_size,
                 input_size, input_dim, cnn1_dim, cnn2_dim):

        self.device = device
        self.input_size = input_size
        
        self.encoder = Encoder(input_dim=input_dim, 
                               cnn1_dim=cnn1_dim, 
                               cnn2_dim=cnn2_dim,
                               ).to(self.device)
        self.decoder = Decoder(input_dim=input_dim,
                               cnn1_dim=cnn1_dim,
                               cnn2_dim=cnn2_dim,
                               ).to(self.device)
        self.params_to_optimize = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        ### Define the loss, optimizer, and training parameters
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.params_to_optimize, lr=learning_rate)
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        # Move both the encoder and the decoder to the selected device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def get_encoded(self, observation, reconstruction=True):
        observation = np.reshape(observation, (1,2,self.input_size,self.input_size))
        observation = torch.from_numpy(observation).float().to(self.device)
        encoded = self.encoder(observation)
        if reconstruction:
            decoded = self.decoder(encoded)
            return torch.squeeze(encoded), torch.squeeze(decoded).cpu().detach().numpy()
        else:
            return torch.squeeze(encoded)

    def train(self, buffer):
        if buffer.buffer_size < self.batch_size:
            return None
        # Set train mode for both the encoder and the decoder
        self.encoder.train()
        self.decoder.train()
        running_loss = 0

        batch_idx = np.random.choice(buffer.buffer_size, self.batch_size, replace=False)
        dataloader = DataLoader(batch_idx, batch_size=self.minibatch_size, shuffle=True)
        for _,minibatch_idx in enumerate(dataloader):
            observations = torch.from_numpy(buffer.observation[minibatch_idx])
            observations = np.reshape(
                observations,
                (observations.size(0), 2, self.input_size, self.input_size)
                ).float().to(self.device)
            encoded = self.encoder(observations)
            decoded = self.decoder(encoded)
            mse_loss = self.loss(decoded, observations)

            loss = mse_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
    
        epoch_loss = running_loss
        return epoch_loss
    
    def load(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name+'_autoencoder.pt', map_location=torch.device(self.device))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        print('Autoencoder loaded')

    def save(self, checkpoint_name):
        torch.save({'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    }, checkpoint_name+'_autoencoder.pt')
        print('Autoencoder saved')



#####  VERGENCE CONTROLLER  #####

class VergenceCritic(nn.Module):
    def __init__(self, learning_rate, n_actions, n_sac_actions,
                 input_size, input_dim, cnn_dim, fc_dim):
        super().__init__()

        self. convpool = nn.Sequential(
            nn.Conv2d(input_dim, cnn_dim, kernel_size=(2,2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.flatten = nn.Flatten(start_dim=-3)
        self.fc = nn.Sequential(
            nn.Linear(2 * int((input_size-1)/2)**2 * cnn_dim + n_sac_actions + n_actions, fc_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(fc_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, encoding_fine, encoding_coarse, sac_action, action):
        encoding_fine = self.convpool(encoding_fine)
        encoding_fine = self.flatten(encoding_fine)
        encoding_coarse = self.convpool(encoding_coarse)
        encoding_coarse = self.flatten(encoding_coarse)
        x = torch.cat([encoding_fine, encoding_coarse, sac_action, action], dim=-1)
        x = self.fc(x)
        critic = self.critic_head(x)
        return critic

class VergenceActor(nn.Module):
    def __init__(self, learning_rate, max_action, n_actions, n_sac_actions,
                 input_size, input_dim, cnn_dim, fc_dim):
        super().__init__()

        self.max_action = max_action
        self.reparam_noise = 1e-6

        self. convpool = nn.Sequential(
            nn.Conv2d(input_dim, cnn_dim, kernel_size=(2,2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2)
        )
        self.flatten = nn.Flatten(start_dim=-3)
        self.fc = nn.Sequential(
            nn.Linear(2 * int((input_size-1)/2)**2 * cnn_dim + n_sac_actions, fc_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(fc_dim, n_actions)
        self.sigma = nn.Linear(fc_dim, n_actions)  

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  

    def forward(self, encoding_fine, encoding_coarse, sac_action):
        encoding_fine = self.convpool(encoding_fine)
        encoding_fine = self.flatten(encoding_fine)
        encoding_coarse = self.convpool(encoding_coarse)
        encoding_coarse = self.flatten(encoding_coarse)
        x = torch.cat([encoding_fine, encoding_coarse, sac_action], dim=-1)
        x = self.fc(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample(self, encoding_fine, encoding_coarse, sac_action,
               reparameterize=True, deterministic=False):
        '''
        Sample action using outputs from actor network
        '''
        mu, sigma = self.forward(encoding_fine, encoding_coarse, sac_action)
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
        log_probs = log_probs.sum(-1, keepdim=True)
        action = action * torch.tensor(self.max_action)

        return action, log_probs


class VergenceController():

    def __init__(self, device, input_size, input_dim, cnn_dim, fc_dim,
                 batch_size, minibatch_size, learning_rate, 
                 n_actions, n_sac_actions, max_action,
                 gamma, alpha, epsilon,
                 ):

        self.device = device
        self.actor = VergenceActor(
                           learning_rate=learning_rate,
                           max_action=max_action,
                           n_actions=n_actions,
                           n_sac_actions=n_sac_actions,
                           input_size=input_size,
                           input_dim=input_dim,
                           cnn_dim=cnn_dim,
                           fc_dim=fc_dim,
                           ).to(self.device)
        self.critic1 = VergenceCritic(
                           learning_rate=learning_rate,
                           n_actions=n_actions,
                           n_sac_actions=n_sac_actions,
                           input_size=input_size,
                           input_dim=input_dim,
                           cnn_dim=cnn_dim,
                           fc_dim=fc_dim,
                           ).to(self.device)
        self.critic2 = VergenceCritic(
                           learning_rate=learning_rate,
                           n_actions=n_actions,
                           n_sac_actions=n_sac_actions,
                           input_size=input_size,
                           input_dim=input_dim,
                           cnn_dim=cnn_dim,
                           fc_dim=fc_dim,
                           ).to(self.device)
        self.max_action = max_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size=batch_size
        self.minibatch_size=minibatch_size
        
    def choose_action(self, encoding_fine, encoding_coarse, sac_action, deterministic=False):
        '''
        Choose action with two-scale encodings using actor network.
        '''
        if deterministic==False and np.random.uniform() < self.epsilon:
            action = np.random.uniform(low=-self.max_action, high=self.max_action)
        else:
            encoding_fine = encoding_fine.to(self.device)
            encoding_coarse = encoding_coarse.to(self.device)
            actions, _ = self.actor.sample(encoding_fine, encoding_coarse, sac_action,
                            reparameterize=False, deterministic=deterministic)
            action = actions.cpu().detach().numpy()[0]
        return np.reshape(action, (1))

    def train(self, buffer, deterministic=False):
        '''
        Train actor and critic networks on a batch sampled from buffer.
        '''
        if buffer.buffer_size < self.batch_size:
            return 0

        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        running_loss = 0
        batch_idx = np.random.choice(buffer.buffer_size, self.batch_size, replace=False)
        dataloader = DataLoader(batch_idx, batch_size=self.minibatch_size, shuffle=True)
        
        for _,minibatch_idx in enumerate(dataloader):
            encoding_fine = torch.from_numpy(buffer.encoding_fine[minibatch_idx]).float().to(self.device)
            encoding_coarse = torch.from_numpy(buffer.encoding_coarse[minibatch_idx]).float().to(self.device)
            sac_action = torch.from_numpy(buffer.sac_action[minibatch_idx]).float().to(self.device)
            reward = torch.from_numpy(buffer.reward[minibatch_idx]).float().to(self.device)
            action = torch.from_numpy(buffer.vrg_action[minibatch_idx]).float().to(self.device)
            done = torch.from_numpy(buffer.done[minibatch_idx]).float().to(self.device)
            
            # Train critic networks
            new_actions, new_log_probs = self.actor.sample(encoding_fine, encoding_coarse, sac_action,
                                                reparameterize=True, deterministic=deterministic)
            new_log_probs = new_log_probs.view(-1)
            critic_1_new_policy = self.critic1.forward(encoding_fine, encoding_coarse, sac_action, new_actions)
            critic_2_new_policy = self.critic2.forward(encoding_fine, encoding_coarse, sac_action, new_actions)
            new_critic_value = torch.min(critic_1_new_policy, critic_2_new_policy)
            new_critic_value = new_critic_value.view(-1)
            
            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            critic_target = reward + self.gamma * (1-done) * (new_critic_value - self.alpha*new_log_probs)
            critic_1_old_policy = self.critic1.forward(encoding_fine, encoding_coarse, sac_action, action).view(-1)
            critic_2_old_policy = self.critic2.forward(encoding_fine, encoding_coarse, sac_action, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(critic_1_old_policy.float(), critic_target.float())
            critic_2_loss = 0.5 * F.mse_loss(critic_2_old_policy.float(), critic_target.float())
            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward(retain_graph=True)
            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            # Train actor network
            new_actions, new_log_probs = self.actor.sample(encoding_fine, encoding_coarse, sac_action,
                                                reparameterize=True, deterministic=deterministic)
            new_log_probs = new_log_probs.view(-1)
            critic_1_new_policy = self.critic1.forward(encoding_fine, encoding_coarse, sac_action, new_actions)
            critic_2_new_policy = self.critic2.forward(encoding_fine, encoding_coarse, sac_action, new_actions)
            new_critic_value = torch.min(critic_1_new_policy, critic_2_new_policy)
            new_critic_value = new_critic_value.view(-1)
            
            self.actor.optimizer.zero_grad()
            actor_loss = self.alpha*new_log_probs - new_critic_value
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            running_loss += critic_loss.item() + actor_loss.item()

        return running_loss
        
    def load(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name+'_vergence.pt', map_location=torch.device(self.device))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        print('Vergence controller loaded')

    def save(self, checkpoint_name):
        torch.save({'actor': self.actor.state_dict(),
                    'critic1': self.critic1.state_dict(),
                    'critic2': self.critic2.state_dict(),
                    }, checkpoint_name+'_vergence.pt')
        print('Vergence controller saved')


##### AGENT #####

class Agent():

    def __init__(self, 
                 # AUTOENCODER PARAMETERS
                 ae_input_size=64, ae_learning_rate=1e-3,
                 ae_batch_size=1000, ae_minibatch_size=100,
                 ae_input_dim=2, ae_cnn1_dim=96, ae_cnn2_dim=24,
                 # VERGENCE PARAMETERS
                 vrg_input_size=7, vrg_input_dim=24, vrg_cnn_dim=24, vrg_fc_dim=1024,
                 vrg_n_actions=1, vrg_max_action=1, vrg_learning_rate=3e-4, vrg_gamma=0.99, 
                 vrg_alpha=0.2, vrg_epsilon=0.05, vrg_batch_size=1000, vrg_minibatch_size=100,
                 sac_n_actions=2,
                ):
        
        self.device = get_device()

        self.autoencoder = Autoencoder(input_size=ae_input_size,
                                        device=self.device,
                                        learning_rate=ae_learning_rate,
                                        batch_size=ae_batch_size,
                                        minibatch_size=ae_minibatch_size,
                                        input_dim=ae_input_dim,
                                        cnn1_dim=ae_cnn1_dim,
                                        cnn2_dim=ae_cnn2_dim,
                                        )
        
        self.vergence = VergenceController(device=self.device,
                                        input_size=vrg_input_size,
                                        input_dim=vrg_input_dim,
                                        cnn_dim=vrg_cnn_dim,
                                        fc_dim=vrg_fc_dim,
                                        batch_size=vrg_batch_size,
                                        minibatch_size=vrg_minibatch_size,
                                        learning_rate=vrg_learning_rate,
                                        n_actions=vrg_n_actions,
                                        n_sac_actions=sac_n_actions,
                                        max_action=vrg_max_action,
                                        gamma=vrg_gamma,
                                        alpha=vrg_alpha,
                                        epsilon=vrg_epsilon,
                                        )