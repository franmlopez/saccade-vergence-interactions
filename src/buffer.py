import numpy as np

class Buffer():

    def __init__(self, max_buffer_size=1000):
        self.buffer_counter = 0
        self.buffer_size = 0
        self.max_buffer_size = max_buffer_size

class AutoencoderBuffer(Buffer):

    def __init__(self):
        super().__init__()
        self.max_buffer_size = 2*self.max_buffer_size
        self.observation = np.array([])

    def store(self, observation):
        if self.buffer_counter == 0:
            self.observation = np.array([observation])
        else:
            self.observation = np.concatenate((self.observation, [observation]), axis=0)
            self.observation = self.observation[-self.max_buffer_size:]
        self.buffer_counter += 1
        self.buffer_size = np.min([self.buffer_counter, self.max_buffer_size])


class VergenceBuffer(Buffer):

    def __init__(self):
        super().__init__()
        self.encoding_fine = np.array([])
        self.encoding_coarse = np.array([])
        self.sac_action = np.array([])
        self.vrg_action = np.array([])
        self.done = np.array([])
        self.reward = np.array([])
    
    def store(self, encoding_fine, encoding_coarse, sac_action, vrg_action, done, reward):
        if self.buffer_counter == 0:
            self.encoding_fine = np.array([encoding_fine.cpu().detach().numpy()])
            self.encoding_coarse = np.array([encoding_coarse.cpu().detach().numpy()])
            self.sac_action = np.array([sac_action])
            self.vrg_action = np.array([vrg_action])
            self.done = np.array([done])
            self.reward = np.array([reward])
        else:
            self.encoding_fine = np.concatenate((self.encoding_fine, [encoding_fine.cpu().detach().numpy()]), axis=0)
            self.encoding_fine = self.encoding_fine[-self.max_buffer_size:]
            self.encoding_coarse = np.concatenate((self.encoding_coarse, [encoding_coarse.cpu().detach().numpy()]), axis=0)
            self.encoding_coarse = self.encoding_coarse[-self.max_buffer_size:]
            self.sac_action = np.concatenate((self.sac_action, [sac_action]), axis=0)
            self.sac_action = self.sac_action[-self.max_buffer_size:]
            self.vrg_action = np.concatenate((self.vrg_action, [vrg_action]), axis=0)
            self.vrg_action = self.vrg_action[-self.max_buffer_size:]
            self.done = np.concatenate((self.done, [done]), axis=0)
            self.done = self.done[-self.max_buffer_size:]
            self.reward = np.concatenate((self.reward, [reward]), axis=0)
            self.reward = self.reward[-self.max_buffer_size:]
        self.buffer_counter += 1
        self.buffer_size = np.min([self.buffer_counter, self.max_buffer_size])
