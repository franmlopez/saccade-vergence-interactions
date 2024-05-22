import numpy as np

class Buffer():

    def __init__(self, max_buffer_size=1000):
        self.buffer_counter = 0
        self.buffer_size = 0
        self.max_buffer_size = max_buffer_size
        self.observation_fine = np.array([])
        self.observation_coarse = np.array([])
        self.old_observation_fine = np.array([])
        self.old_observation_coarse = np.array([])
        self.encoding_fine = np.array([])
        self.encoding_coarse = np.array([])
        self.magno_encoding_coarse = np.array([])
        self.magno_encoding_fine = np.array([])
        self.action = np.array([])
        self.done = np.array([])
        self.vergence_reward = np.array([])
        self.cyclo_reward = np.array([])
        self.version_reward = np.array([])
        self.accommodation_reward = np.array([])
        

    def store(self, observation_fine, observation_coarse, old_observation_fine, old_observation_coarse, 
              encoding_fine, encoding_coarse, magno_encoding_fine, magno_encoding_coarse, action,done,
              vergence_reward, cyclo_reward, version_reward, accommodation_reward):
        '''
        Append a single experience to end of buffer.
        '''
        if self.buffer_counter == 0:
            self.observation_fine = np.array([observation_fine])
            self.observation_coarse = np.array([observation_coarse])
            self.old_observation_fine = np.array([old_observation_fine])
            self.old_observation_coarse = np.array([old_observation_coarse])
            self.encoding_fine = np.array([encoding_fine.detach().numpy()])
            self.encoding_coarse = np.array([encoding_coarse.detach().numpy()])
            self.magno_encoding_fine = np.array([magno_encoding_fine.detach().numpy()])
            self.magno_encoding_coarse = np.array([magno_encoding_coarse.detach().numpy()])
            self.action = np.array([action])
            self.done = np.array([done])
            self.vergence_reward = np.array([vergence_reward])
            self.cyclo_reward = np.array([cyclo_reward])
            self.version_reward = np.array([version_reward])
            self.accommodation_reward = np.array([accommodation_reward])
        else: 
            self.observation_fine = np.concatenate((self.observation_fine, [observation_fine]), axis=0)
            self.observation_coarse = np.concatenate((self.observation_coarse, [observation_coarse]), axis=0)
            self.old_observation_fine = np.concatenate((self.old_observation_fine, [old_observation_fine]), axis=0)
            self.old_observation_coarse = np.concatenate((self.old_observation_coarse, [old_observation_coarse]), axis=0)
            self.encoding_fine = np.concatenate((self.encoding_fine, [encoding_fine.detach().numpy()]), axis=0)
            self.encoding_coarse = np.concatenate((self.encoding_coarse, [encoding_coarse.detach().numpy()]), axis=0)
            self.magno_encoding_fine = np.concatenate((self.magno_encoding_fine, [magno_encoding_fine.detach().numpy()]), axis=0)
            self.magno_encoding_coarse = np.concatenate((self.magno_encoding_coarse, [magno_encoding_coarse.detach().numpy()]), axis=0)
            self.action = np.concatenate((self.action, [action]), axis=0)
            self.done = np.concatenate((self.done, [done]), axis=0)
            self.vergence_reward = np.concatenate((self.vergence_reward, [vergence_reward]), axis=0)
            self.cyclo_reward = np.concatenate((self.cyclo_reward, [cyclo_reward]), axis=0)
            self.version_reward = np.concatenate((self.version_reward, [version_reward]), axis=0)
            self.accommodation_reward = np.concatenate((self.accommodation_reward, [accommodation_reward]), axis=0)
        self.buffer_counter += 1

    def concat(self, process_buffer):
        '''
        Concatenate a process buffer to a global buffer.
        '''
        if self.buffer_counter == 0:
            self.observation_fine = np.array(process_buffer.observation_fine)
            self.observation_coarse = np.array(process_buffer.observation_coarse)
            self.old_observation_fine = np.array(process_buffer.old_observation_fine)
            self.old_observation_coarse = np.array(process_buffer.old_observation_coarse)
            self.encoding_fine = np.array(process_buffer.encoding_fine)
            self.encoding_coarse = np.array(process_buffer.encoding_coarse)
            self.magno_encoding_coarse = np.array(process_buffer.magno_encoding_coarse)
            self.magno_encoding_fine = np.array(process_buffer.magno_encoding_fine)
            self.action = np.array(process_buffer.action)
            self.done = np.array(process_buffer.done)
            self.vergence_reward = np.array(process_buffer.vergence_reward)
            self.cyclo_reward = np.array(process_buffer.cyclo_reward)
            self.version_reward = np.array(process_buffer.version_reward)
            self.accommodation_reward = np.array(process_buffer.accommodation_reward)
        else:
            # concatenate process buffer to end of global buffer:
            self.observation_fine = np.concatenate((self.observation_fine, process_buffer.observation_fine), axis=0)
            self.observation_coarse = np.concatenate((self.observation_coarse, process_buffer.observation_coarse), axis=0)
            self.old_observation_fine = np.concatenate((self.old_observation_fine, process_buffer.old_observation_fine), axis=0)
            self.old_observation_coarse = np.concatenate((self.old_observation_coarse, process_buffer.old_observation_coarse), axis=0)
            self.encoding_fine = np.concatenate((self.encoding_fine, process_buffer.encoding_fine), axis=0)
            self.encoding_coarse = np.concatenate((self.encoding_coarse, process_buffer.encoding_coarse), axis=0)
            self.magno_encoding_coarse = np.concatenate((self.magno_encoding_coarse, process_buffer.magno_encoding_coarse), axis=0)
            self.magno_encoding_fine = np.concatenate((self.magno_encoding_fine, process_buffer.magno_encoding_fine), axis=0)
            self.action = np.concatenate((self.action, process_buffer.action), axis=0)
            self.done = np.concatenate((self.done, process_buffer.done), axis=0)
            self.vergence_reward = np.concatenate((self.vergence_reward, process_buffer.vergence_reward), axis=0)
            self.cyclo_reward = np.concatenate((self.cyclo_reward, process_buffer.cyclo_reward), axis=0)
            self.version_reward = np.concatenate((self.version_reward, process_buffer.version_reward), axis=0)
            self.accommodation_reward = np.concatenate((self.accommodation_reward, process_buffer.accommodation_reward), axis=0)
            # clip global buffer to max_buffer_size:
            self.observation_fine = self.observation_fine[-self.max_buffer_size:]
            self.observation_coarse = self.observation_coarse[-self.max_buffer_size:]
            self.old_observation_fine = self.old_observation_fine[-self.max_buffer_size:]
            self.old_observation_coarse = self.old_observation_coarse[-self.max_buffer_size:]
            self.encoding_fine = self.encoding_fine[-self.max_buffer_size:]
            self.encoding_coarse = self.encoding_coarse[-self.max_buffer_size:]
            self.magno_encoding_coarse = self.magno_encoding_coarse[-self.max_buffer_size:]
            self.magno_encoding_fine = self.magno_encoding_fine[-self.max_buffer_size:]
            self.action = self.action[-self.max_buffer_size:]
            self.done = self.done[-self.max_buffer_size:]
            self.vergence_reward = self.vergence_reward[-self.max_buffer_size:]
            self.cyclo_reward =self.cyclo_reward[-self.max_buffer_size:]
            self.version_reward =self.version_reward[-self.max_buffer_size:]
            self.accommodation_reward =self.accommodation_reward[-self.max_buffer_size:]
        self.buffer_counter += process_buffer.buffer_counter
        self.buffer_size = np.min([self.buffer_counter, self.max_buffer_size])