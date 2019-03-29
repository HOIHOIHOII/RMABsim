# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:54:32 2019

@author: AI
"""

#to see if the single armed bandit sim has an average pull value that 
#converges quickly

from __future__ import division
import numpy as np
import pandas as pd
import random


random.seed(1)

#state transition matrix
#all rowsums = 1
stm = np.array([
               [0.05,0.05,0.9]
              ,[0.4,0.01,0.59]
              ,[0.8,0.1,0.1]
              ])

shape = stm.shape
#write down reward vector
rv = np.array([1,2,0])

#define initial state and reward
initial_state_and_reward = [["State","Reward"], 
                            [1,0]
                           ]

#generate one step and reward 
def pull_lever(current_state_and_reward):
    '''takes the current state and generates the next state and its reward, 
    assumes stm and rv defined globally'''
    state, _ = current_state_and_reward
    n = shape[0] #number of states
    #weighted random choice of next state from 
    #current using state transition matrix
    next_state = np.random.choice(range(0,n),p = stm[state,:])
    reward = rv[next_state]
    return [next_state, reward]

#generate a full set of steps over one trial
def run_trial(initial_state, k=100):
    '''takes an initial state DF and returns a DF with k trials'''
    data = initial_state[:] 
    #python weirdness: very important to take slice, 
    #                  otherwise just get pointer to original, which will
    #                  in this instance modify var in global scope
    for i in range(k):
        result = pull_lever(data[-1])
        data += [result]
    return data

def compute_pull_avg(trial):
    initial_data = np.array(trial[1:])
    n = initial_data.shape[0]
    reward = initial_data[:,1].sum()
    print initial_data[:,1]
    print n
    return reward/n

run_trial(initial_state_and_reward,100)

compute_pull_avg(run_trial(initial_state_and_reward,100000))

#generate many trials and record all

random.random()

(1.25/2.91*1)+(0.16/2.91*2)
