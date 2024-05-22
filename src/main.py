import time
import copy
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter

import os, sys
file_path = os.path.abspath(os.path.dirname(__file__))
main_path = os.path.abspath(os.path.join(file_path,'..'))
if main_path != sys.path[0]:
    sys.path.insert(0, main_path)
    os.chdir(main_path)
    del main_path, file_path

from src.embodiment import Embodiment
from src.utils import make_dir, write_values

def train(folder_name, checkpoint_name, checkpoint_num,
          n_epochs, train_every, save_every,
          sac_vrg_int, vrg_reward_method):
    '''
    Runs a full experiment with the parameters defined in config.
    '''

    make_dir(folder_name)
    writer = SummaryWriter(folder_name)

    embodiment = Embodiment(checkpoint_name=checkpoint_name, 
                            checkpoint_num=checkpoint_num,
                            sac_vrg_int=sac_vrg_int,
                            vrg_reward_method=vrg_reward_method,
                           )
    
    start = time.time()

    for epoch in tqdm(range(n_epochs+1)):
        start_epoch = time.time()

        for _ in range(train_every):
            embodiment.run()

        # Train autoencoder and actor-critic from a buffer of experiences
        ae_loss = embodiment.agent.autoencoder.train(embodiment.ae_buffer)
        vergence_loss = embodiment.agent.vergence.train(embodiment.vrg_buffer)
        
        if (epoch%save_every==0):
            embodiment.save_agent(checkpoint_name=folder_name, checkpoint_num=epoch)

        # Outputs:
        end_epoch = time.time()
        vergence_reward = embodiment.ep_vergence_reward
        distance = embodiment.ep_distance
        disparity = embodiment.ep_disparity
        writer.add_scalar("loss/autoencoder", (ae_loss if ae_loss is not None else np.nan), epoch)
        writer.add_scalar("loss/vergence", (vergence_loss if vergence_loss is not None else np.nan), epoch)
        writer.add_scalar("time/epoch_time", end_epoch - start_epoch, epoch)
        writer.add_scalar("metric/vergence_reward", (vergence_reward if vergence_reward is not None else np.nan), epoch)
        writer.add_scalar("metric/distance", (distance if distance is not None else np.nan), epoch)
        writer.add_scalar("metric/disparity", (disparity if disparity is not None else np.nan), epoch)

    writer.close()
    end = time.time()
    print("\nCompleted experiment in %d seconds" %(end-start))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', default=None, type=str)
    parser.add_argument('--checkpoint_num', default=None, type=int)
    parser.add_argument('--checkpoint_name', default=None, type=str)
    parser.add_argument('--n_epochs', default=10000, type=int)
    parser.add_argument('--save_every', default=1000, type=int)
    parser.add_argument('--train_every', default=5, type=int)
    parser.add_argument('--svi', default=True, type=bool)
    parser.add_argument('--reward_method', default='lopez', type=str,
                        choices=['lopez','wilmot','zhu'])
    

    args = parser.parse_args()
    folder_name = args.folder_name

    if folder_name is not None:
        folder_name = 'results/' + folder_name
    else:
        folder_name = 'results/' + time.strftime("%Y%m%d")
    

    train(
        folder_name=folder_name,
        checkpoint_name=args.checkpoint_name,
        checkpoint_num=args.checkpoint_num,
        n_epochs=args.n_epochs,
        save_every=args.save_every,
        train_every=args.train_every,
        sac_vrg_int=args.svi,
        vrg_reward_method=args.reward_method,
    )
