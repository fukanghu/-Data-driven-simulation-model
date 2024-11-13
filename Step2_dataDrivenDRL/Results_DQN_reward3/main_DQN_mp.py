# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""

import sys 
sys.path.append("..") 

import ENV
import numpy as np
import pandas as pd
from DQN import Buffer
from DQN import DQN as DQN
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
import time

Train=True
init_train=True

tf.compat.v1.reset_default_graph()

data = pd.read_csv('../DATA/rdata.csv').iloc[:,1:]
index = np.load('../DATA/index.npy',allow_pickle=True).tolist()
action_table = np.load('../DATA/action_table1.npy',allow_pickle=True).tolist()

step = 24

agent_params={
    'state_dim':len(index['state']),
    'action_dim':len(action_table),

    'encoding_layer':[50,50],
    'value_layer':[50,50],
    'advantage_layer':[50,50],
    'test_num':100, # see pretrain_data

    'train_iterations':2,
    'training_step':100,
    'gamma':0.01,
    'epsilon':0.1,
    'ep_min':1e-50,
    'ep_decay':0.9,
    'learning_rate':0.001,

    'action_table':np.array(action_table),
}

model = DQN.DQN(agent_params)
if init_train:
    model.model.save_weights('./model/dqn.h5')    
    model.target_model.save_weights('./model/target_dqn.h5')    
model.load_model('./model/')
print('model done')


###############################################################################
# Train
###############################################################################


def interact(timepoint,step,controlmodel):
    env = ENV.ENV()
    results = env.reset(timepoint,step)
    for s in range(step-1):
        observation = np.array(data.values[env.timepoint + 1 + step,env.title['state']+env.title['inflow']].tolist()).reshape((1,-1))
        a = DQN.sample_action(observation,controlmodel,False)
        #_, a = PPO.sample_action(observation,ppomodel,False)
        action = action_table[a.numpy()[0]]
        results, done = env.step(action)
    
    return results



if Train:
    #tf.config.experimental_run_functions_eagerly(True)
    t1 = time.perf_counter()
    t2=[]

    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = Buffer.Buffer(model.params['state_dim'], int(data.shape[0])*model.params['test_num'])
        
        # Iterate over the steps of each epoch
        res = []
        for timepoint in range(step,model.params['test_num']+step-3):
            episode_return, episode_length = 0, 0
            optdata = interact(timepoint,step,model)
            o=optdata['state'][-1][1:-2]
            a=optdata['action'][-1][1:-1]
            r=optdata['rewards'][1:-2]
            o_=optdata['state'][-1][2:-1]
            last_value=0
            episode_return += np.sum(r)
            episode_length += len(r)
            res.append([o,a,r,o_,last_value,episode_return,episode_length])

        for i in range(model.params['test_num']-3):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,o_ in zip(res[i][0],res[i][1],res[i][2],res[i][3]):
                #buffer.store(o,a,r,o_) # For OPT data
                #For EFD data
                atem = 0
                for it in range(agent_params['action_table'].shape[0]):
                    if (agent_params['action_table'][it] == np.array(a)).all():
                        atem = it
                buffer.store(o,atem,r,o_)
            buffer.finish_trajectory(res[i][4])
            sum_return += res[i][5]
            sum_length += res[i][6]
            num_episodes += 1
        
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            observation_next_buffer,
            reward_buffer,
            advantage_buffer,
        ) = buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(model.params['train_iterations']):
            l=DQN.train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model)
            history['Loss'].append(l)
            
        model.model.save_weights('./model/dqn.h5')
        model.model.save_weights('./model/target_dqn.h5')    
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 10 == 0:
            model.params['epsilon'] *= model.params['ep_decay']
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Loss: {history['Loss'][-1]}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}. Epsilon: {model.params['epsilon']}"
        )
        
        np.save('./Results/Train1.npy',history)
        if epoch == 50 or epoch == 100 or epoch == 1000:
            t2.append(time.perf_counter())

    timeR=[ti-t1 for ti in t2]
    print('Running time: ',timeR)
    np.save('./time.npy',timeR)
    
    # plot
    #plt.figure()
    #plt.plot(history['Loss'])
    #plt.savefig('./Results/Train1.tif')

   
###############################################################################
# end Train
###############################################################################
