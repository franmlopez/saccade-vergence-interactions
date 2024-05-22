import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models import Autoencoder, MagnoAutoencoder, Actor, Critic, sparse_loss
from utils.buffer import Buffer
from utils.auxiliary import get_device

from src.config import vergence, cyclo, version, accommodation, \
                        ae_learning_rate, ae_sparse_lambda, ac_learning_rate, gamma, alpha, \
                        epsilon, batch_size, minibatch_size,\
                        parvo_ae_input_dim, parvo_ae_cnn1_dim, parvo_ae_cnn2_dim, \
                        magno_ae_input_dim, magno_ae_cnn1_dim, magno_ae_cnn2_dim, \
                        max_vergence_action, max_version_action


class Agent():
    def __init__(self, img_size=32, max_vergence_action=max_vergence_action, max_version_action=max_version_action,
                ae_learning_rate=ae_learning_rate, ae_sparse_lambda=ae_sparse_lambda,
                gamma=gamma, alpha=alpha, ac_learning_rate=ac_learning_rate, 
                minibatch_size=minibatch_size, batch_size=batch_size, epsilon=epsilon,
                vergence=vergence, cyclo=cyclo, 
                accommodation=accommodation, version=version,
                ):
        self.device = 'cpu' #get_device()
        self.img_size = img_size    # ASSIGN TO ENVIRONMENT
        # autoencoder
        self.autoencoder = Autoencoder().float()
        self.ae_sparse_lambda = ae_sparse_lambda    # for sparsity of encodings
        self.ae_learning_rate = ae_learning_rate
        self.ae_criterion = nn.MSELoss()
        self.ae_optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.ae_learning_rate)
        # magnocellular autoencoder
        self.magno_autoencoder = MagnoAutoencoder().float()
        self.magno_ae_criterion = nn.MSELoss()
        self.magno_ae_optimizer = optim.Adam(self.magno_autoencoder.parameters(), lr=self.ae_learning_rate)
        # reinforcement learning
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration/exploitation trade-off
        self.alpha = alpha    # entropy coefficient trade-off (cf. SAC paper)
        self.max_vergence_action = max_vergence_action    # max angle movement per timestep
        self.max_version_action = max_version_action
        self.minibatch_size = minibatch_size
        self.batch_size = batch_size
        self.buffer = Buffer()
        # vergence
        if vergence:
            self.vergence = True
            self.vergence_actor = Actor(ac_learning_rate,n_actions=1, max_action=self.max_vergence_action,
                                        input_dim=parvo_ae_cnn2_dim, cnn_dim=parvo_ae_cnn2_dim).to(self.device)
            self.vergence_critic_1 = Critic(ac_learning_rate, n_actions=1,    # as per SAC paper, using 
                                        input_dim=parvo_ae_cnn2_dim, cnn_dim=parvo_ae_cnn2_dim).to(self.device)
            self.vergence_critic_2 = Critic(ac_learning_rate, n_actions=1,
                                        input_dim=parvo_ae_cnn2_dim, cnn_dim=parvo_ae_cnn2_dim).to(self.device)
        else:
            self.vergence=False
        # cycloverge
        if cyclo:
            self.cyclo = True
            self.cyclo_actor = Actor(ac_learning_rate,n_actions=1, max_action=self.max_vergence_action).to(self.device)
            self.cyclo_critic_1 = Critic(ac_learning_rate, n_actions=1).to(self.device)
            self.cyclo_critic_2 = Critic(ac_learning_rate, n_actions=1).to(self.device)
        else:
            self.cyclo = False
        # accommodation
        if accommodation:
            self.accommodation = True
            self.accommodation_actor = Actor(ac_learning_rate, n_actions=1, max_action=self.max_vergence_action).to(self.device)
            self.accommodation_critic_1 = Critic(ac_learning_rate, n_actions=1).to(self.device)
            self.accommodation_critic_2 = Critic(ac_learning_rate, n_actions=1).to(self.device)
        else:
            self.accommodation = False
        # version
        if version:
            self.version = True
            self.version_actor = Actor(ac_learning_rate,n_actions=2, max_action=self.max_version_action,
                                        input_dim=magno_ae_cnn2_dim, cnn_dim=magno_ae_cnn2_dim).to(self.device)
            self.version_critic_1 = Critic(ac_learning_rate, n_actions=2,
                                        input_dim=magno_ae_cnn2_dim, cnn_dim=magno_ae_cnn2_dim).to(self.device)
            self.version_critic_2 = Critic(ac_learning_rate, n_actions=2,
                                        input_dim=magno_ae_cnn2_dim, cnn_dim=magno_ae_cnn2_dim).to(self.device)
        else:
            self.version = False

    # --- Autoencoder ---

    # get one encoding and reconstruction loss:
    def get_parvo_encoding(self, observation):
        '''
        Returns encoding and reconstruction loss from autoencoder.
        '''
        observation = np.reshape(observation, (1,2,32,32))
        observation = torch.from_numpy(observation).float().to(self.device)
        encoding, reconstruction = self.autoencoder(observation)
        encoding = torch.reshape(encoding, (parvo_ae_cnn2_dim,7,7))
        loss = self.ae_criterion(reconstruction, observation).item()
        return encoding, loss

    def get_magno_encoding(self, observation, old_observation):
        '''
        Returns encoding and reconstruction loss from autoencoder.
        '''
        observation = torch.from_numpy(np.reshape(observation, (1,2,32,32))).float()
        old_observation = torch.from_numpy(np.reshape(old_observation, (1,2,32,32))).float()
        observation = torch.cat([observation, old_observation], dim=1).to(self.device)
        encoding, reconstruction = self.magno_autoencoder(observation)
        encoding = torch.reshape(encoding, (magno_ae_cnn2_dim,7,7))
        loss = self.magno_ae_criterion(reconstruction, observation).item()
        return encoding, loss

    def train_autoencoder(self, buffer):
        '''
        Train autoencoder on a batch sampled from buffer.
        '''
        if buffer.buffer_size < self.batch_size:
            return None
           
        self.autoencoder.train()
        running_loss = 0

        batch_idx = np.random.choice(buffer.buffer_size, self.batch_size, replace=False)
        dataloader = DataLoader(batch_idx, batch_size=self.minibatch_size, shuffle=True)
        for _,minibatch_idx in enumerate(dataloader):

            observations_fine = torch.from_numpy(buffer.observation_fine[minibatch_idx]).to(self.device)
            observations_fine = np.reshape(observations_fine, (observations_fine.size(0),2,32,32)).float()
            encodings_fine, reconstructions_fine = self.autoencoder(observations_fine)
            mse_loss_fine = self.ae_criterion(reconstructions_fine, observations_fine)
            l1_loss_fine = sparse_loss(encodings_fine)

            observations_coarse = torch.from_numpy(buffer.observation_coarse[minibatch_idx]).to(self.device)
            observations_coarse = np.reshape(observations_coarse, (observations_coarse.size(0),2,32,32)).float()
            encodings_coarse, reconstructions_coarse = self.autoencoder(observations_coarse)
            mse_loss_coarse = self.ae_criterion(reconstructions_coarse, observations_coarse)
            l1_loss_coarse = sparse_loss(encodings_coarse)

            self.ae_optimizer.zero_grad()
            loss = (mse_loss_fine + mse_loss_coarse).mean() + self.ae_sparse_lambda*(l1_loss_fine+l1_loss_coarse).mean()
            loss.backward()
            self.ae_optimizer.step()
            running_loss += loss.item()
    
        epoch_loss = running_loss
        return epoch_loss

    def train_magno_autoencoder(self, buffer):
        '''
        Train autoencoder on a batch sampled from buffer.
        '''
        if buffer.buffer_size < self.batch_size:
            return None
           
        self.magno_autoencoder.train()
        running_loss = 0

        batch_idx = np.random.choice(buffer.buffer_size, self.batch_size, replace=False)
        dataloader = DataLoader(batch_idx, batch_size=self.minibatch_size, shuffle=True)
        for _,minibatch_idx in enumerate(dataloader):

            old_observations_coarse = torch.from_numpy(buffer.old_observation_coarse[minibatch_idx])
            old_observations_coarse = np.reshape(old_observations_coarse, (old_observations_coarse.size(0),2,32,32)).float()
            old_observations_fine = torch.from_numpy(buffer.old_observation_fine[minibatch_idx])
            old_observations_fine = np.reshape(old_observations_fine, (old_observations_fine.size(0),2,32,32)).float()

            observations_coarse = torch.from_numpy(buffer.observation_coarse[minibatch_idx])
            observations_coarse = np.reshape(observations_coarse, (observations_coarse.size(0),2,32,32)).float()
            observations_fine = torch.from_numpy(buffer.observation_fine[minibatch_idx])
            observations_fine = np.reshape(observations_fine, (observations_fine.size(0),2,32,32)).float()

            observations_coarse = torch.cat([observations_coarse, old_observations_coarse], dim=1).to(self.device)
            encodings_coarse, reconstructions_coarse = self.magno_autoencoder(observations_coarse)
            mse_loss_coarse = self.magno_ae_criterion(reconstructions_coarse, observations_coarse)
            l1_loss_coarse = sparse_loss(encodings_coarse)

            observations_fine = torch.cat([observations_fine, old_observations_fine], dim=1).to(self.device)
            encodings_fine, reconstructions_fine = self.magno_autoencoder(observations_fine)
            mse_loss_fine = self.magno_ae_criterion(reconstructions_fine, observations_fine)
            l1_loss_fine = sparse_loss(encodings_fine)
            
            self.magno_ae_optimizer.zero_grad()
            loss = (mse_loss_coarse + mse_loss_fine).mean() + self.ae_sparse_lambda*(l1_loss_coarse+l1_loss_fine).mean()
            loss.backward()
            self.magno_ae_optimizer.step()
            running_loss += loss.item()
    
        epoch_loss = running_loss
        return epoch_loss

    # --- Soft actor critic ---

    def choose_action(self, encoding_1, encoding_2, eye_movement, deterministic=False):
        '''
        Choose action with two-scale encodings using actor network.
        '''
        if deterministic==False and np.random.uniform() < self.epsilon:
            if eye_movement=='vergence':
                return np.array([np.random.uniform(low=-self.max_vergence_action, high=self.max_vergence_action)])
            elif eye_movement=='version':
                return np.random.uniform(low=-self.max_version_action, high=self.max_version_action, size=2)

        else:
            if eye_movement=='vergence':
                encoding_1 = torch.reshape(encoding_1, (1,parvo_ae_cnn2_dim,7,7)).to(self.device)
                encoding_2 = torch.reshape(encoding_2, (1,parvo_ae_cnn2_dim,7,7)).to(self.device)
                actions, _ = self.vergence_actor.sample(encoding_1, encoding_2,
                                            reparameterize=False, deterministic=deterministic)
            if eye_movement=='cyclo':
                actions, _ = self.cyclo_actor.sample(encoding_1, encoding_2,
                                            reparameterize=False, deterministic=deterministic)
            if eye_movement=='version':
                encoding_1 = torch.reshape(encoding_1, (1,magno_ae_cnn2_dim,7,7)).to(self.device)
                encoding_2 = torch.reshape(encoding_2, (1,magno_ae_cnn2_dim,7,7)).to(self.device)
                actions, _ = self.version_actor.sample(encoding_1, encoding_2,
                                            reparameterize=False, deterministic=deterministic)
            return actions.detach().numpy()[0]

    def soft_actor_critic(self, actor, critic_1, critic_2, buffer,
                          eye_movement, deterministic=False):
        '''
        Train actor and critic networks on a batch sampled from buffer.
        '''
        if buffer.buffer_size < self.batch_size:
            return 0

        running_loss = 0
        batch_idx = np.random.choice(buffer.buffer_size, self.batch_size, replace=False)
        dataloader = DataLoader(batch_idx, batch_size=self.minibatch_size, shuffle=True)
        
        for _,minibatch_idx in enumerate(dataloader):
            if eye_movement == 'vergence':
                encoding_1 = torch.from_numpy(buffer.encoding_fine[minibatch_idx]).float()
                encoding_2 = torch.from_numpy(buffer.encoding_coarse[minibatch_idx]).float()
                action = torch.from_numpy(buffer.action[minibatch_idx][:,[0]]).float()
                reward = torch.from_numpy(buffer.vergence_reward[minibatch_idx]).float()
            elif eye_movement == 'version':
                encoding_1 = torch.from_numpy(buffer.magno_encoding_fine[minibatch_idx]).float()
                encoding_2 = torch.from_numpy(buffer.magno_encoding_coarse[minibatch_idx]).float()
                action = torch.from_numpy(buffer.action[minibatch_idx][:,[1,2]]).float()
                reward = torch.from_numpy(buffer.version_reward[minibatch_idx]).float()
            done = torch.from_numpy(buffer.done[minibatch_idx]).float()

            # Train critic networks
            new_actions, new_log_probs = actor.sample(encoding_1, encoding_2,
                                                reparameterize=True, deterministic=deterministic)
            new_log_probs = new_log_probs.view(-1)
            critic_1_new_policy = critic_1.forward(encoding_1, encoding_2, new_actions)
            critic_2_new_policy = critic_2.forward(encoding_1, encoding_2, new_actions)
            new_critic_value = torch.min(critic_1_new_policy, critic_2_new_policy)
            new_critic_value = new_critic_value.view(-1)
            
            critic_1.optimizer.zero_grad()
            critic_2.optimizer.zero_grad()
            critic_target = reward + self.gamma * (1-done) * (new_critic_value - self.alpha*new_log_probs)
            critic_1_old_policy = critic_1.forward(encoding_1, encoding_2, action).view(-1)
            critic_2_old_policy = critic_2.forward(encoding_1, encoding_2, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(critic_1_old_policy.float(), critic_target.float())
            critic_2_loss = 0.5 * F.mse_loss(critic_2_old_policy.float(), critic_target.float())
            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward(retain_graph=True)
            critic_1.optimizer.step()
            critic_2.optimizer.step()

            # Train actor network
            new_actions, new_log_probs = actor.sample(encoding_1, encoding_2,
                                                reparameterize=True, deterministic=deterministic)
            new_log_probs = new_log_probs.view(-1)
            critic_1_new_policy = critic_1.forward(encoding_1, encoding_2, new_actions)
            critic_2_new_policy = critic_2.forward(encoding_1, encoding_2, new_actions)
            new_critic_value = torch.min(critic_1_new_policy, critic_2_new_policy)
            new_critic_value = new_critic_value.view(-1)
            
            actor.optimizer.zero_grad()
            actor_loss = self.alpha*new_log_probs - new_critic_value
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            actor.optimizer.step()

            running_loss += critic_loss.item() + actor_loss.item()

        epoch_loss = running_loss
        return epoch_loss

    def train_vergence_actor_critic(self, buffer, deterministic=False):
        loss = self.soft_actor_critic(actor = self.vergence_actor,
                                critic_1 = self.vergence_critic_1,
                                critic_2 = self.vergence_critic_2,
                                buffer = buffer,
                                eye_movement = 'vergence',
                                deterministic = deterministic)
        return loss

    def train_cyclo_actor_critic(self, buffer, deterministic=False):
        loss = self.soft_actor_critic(actor = self.cyclo_actor,
                                critic_1 = self.cyclo_critic_1,
                                critic_2 = self.cyclo_critic_2,
                                buffer = buffer,
                                eye_movement = 'cyclo',
                                deterministic = deterministic)
        return loss

    def train_version_actor_critic(self, buffer, deterministic=False):
        loss = self.soft_actor_critic(actor = self.version_actor,
                                      critic_1 = self.version_critic_1,
                                      critic_2 = self.version_critic_2,
                                      buffer = buffer,
                                      eye_movement = 'version',
                                      deterministic = deterministic)
        return loss
        
    def train_all_actor_critics(self, buffer, deterministic=False):
        if self.vergence:
            vergence_loss = self.train_vergence_actor_critic(buffer, deterministic=deterministic)
        else:
            vergence_loss = 0
        if self.cyclo:
            cyclo_loss = self.train_cyclo_actor_critic(buffer, deterministic=deterministic)    
        else:
            cyclo_loss = 0
        if self.version:
            version_loss = self.train_version_actor_critic(buffer, deterministic=deterministic)
        else:
            version_loss = 0
        
        total_loss = vergence_loss + cyclo_loss + version_loss
        return total_loss

