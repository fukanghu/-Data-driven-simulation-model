# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:41:39 2022

@author: chong

SWMM environment
can be used for any inp file
established based pyswmm
"""
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import sys
sys.path.insert(0, sys.path[0]+"/../")
sys.path.append("../..") 

import BLSTM_ATT


# 出水达标、控制稳定
def reward(output,action):
    # 用归一化后的state计算reward,size:M,
    #r1 = np.sum(np.abs(state[state_index]-state_bar[state_index])) # 过程稳定, small
    r2 = np.sum(output+1) # 出水达标, small
    r3 = np.sum(np.array(action)+1) # 减少曝气, small
    r = 0.4/r2 + 0.6/r3 
    return r


class ENV:
    #can be used for every SWMM inp
    def __init__(self):
        self.data = pd.read_csv('../DATA/rdata.csv').iloc[:,1:]
        self.action_table = np.load('../DATA/action_table1.npy',allow_pickle=True).tolist()
        self.index = np.load('../DATA/index.npy',allow_pickle=True).tolist()
        
        inflow, control, outflow, state =[], [], [], []
        i = 0
        for k in self.data.columns:
            if '进水' in k:
                inflow.append(i)
            elif 'DO' in k:
                control.append(i)
            elif '出水' in k:
                outflow.append(i)
            else:
                state.append(i)
            i += 1

        self.title = {
            'inflow':inflow, # 进水
            'control':control, # set point of DO
            'state':state, # pool data
            'outflow':outflow,
        }
        self.input_index = self.title['inflow']+self.title['control']+self.title['state']
        self.output_index = self.title['outflow']
        self.timepoint = 0

        # load simulation model
        param=np.load('../model/BLSTM_ATT_param.npy',allow_pickle=True).tolist()
        self.net = BLSTM_ATT.Net(param)
        self.net = torch.load('../model/BLSTM_ATT')
    
    def reset(self,timepoint,step):
        # timepoint:数据中的时间点
        self.starttime = timepoint
        self.timepoint = timepoint
        self.done = False # start step
        self.endstep = step + timepoint # move step
        self.sim_step = 10 # model input time interval
        
        #模拟一步
        x = torch.reshape(Variable(torch.FloatTensor(np.concatenate((self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['inflow']],
                                                self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['control']],
                                                self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['state']]),axis=1))),(1,self.sim_step,-1))# inflow, action, state
        output = self.net(x).detach().numpy()
        
        # save step results
        self.results = {}
        self.results['state'] = [self.data.values[timepoint:timepoint+24,self.title['state']+self.title['inflow']].tolist()]
        self.results['action'] = [self.data.values[timepoint:timepoint+24,self.title['control']].tolist()]
        self.results['rewards'] = [reward(output,self.data.values[timepoint:timepoint+24,self.title['control']].tolist())]
        self.results['outflow'] = [output.tolist()[0]]
        return self.results
        
    def step(self,action):
        
        self.timepoint += 1
        #设置控制
        step_action = self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['control']].copy()
        step_action[-1] = np.array(action)
        #for i in range(-(self.timepoint-self.starttime),-2):
        #    step_action[i] = np.array(self.results['action'])[i]
        x = torch.reshape(Variable(torch.FloatTensor(np.concatenate((self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['inflow']],
                                                step_action,
                                                self.data.values[self.timepoint-10+1:self.timepoint+1,self.title['state']]),axis=1))),(1,self.sim_step,-1))# inflow, action, state
        output = self.net(x).detach().numpy()
        
        #获取reward和结果
        self.results['state'].append(self.data.values[self.timepoint:self.timepoint+24,self.title['state']+self.title['inflow']].tolist())
        self.results['action'].append(action)
        self.results['rewards'].append(reward(output,self.data.values[self.timepoint:self.timepoint+24,self.title['control']].tolist()))
        self.results['outflow'].append(output.tolist()[0])
            
        #降雨结束检测
        if self.timepoint == self.endstep:
            self.done = True
        return self.results,self.done
