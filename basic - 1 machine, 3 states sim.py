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
import matplotlib.pyplot as plt
import random




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
    #python weirdness haiku (by NB): 
    #          Beware the arrow, 
    #          pointing to the original,
    #          slice, but cut nothing
    for i in range(k):
        result = pull_lever(data[-1])
        data += [result]
    return data

#begin script

k=1000

for _ in (1,2):
    random.seed(1)
    
    trial_data = run_trial(initial_state_and_reward,k)
    
    trial_data = pd.DataFrame(data = trial_data[1:], columns = trial_data[0])
    trial_data["Running_Total"] = trial_data.Reward.expanding().sum()
    trial_data["Running_Average"] = trial_data.Reward.expanding().mean()
    
    avg = trial_data.iloc[-1]["Running_Average"]
    print("The long term average over {} trials is: {}".format(k,avg))
    
    fig, (ax1,ax2) = plt.subplots(2,1)
    
    ax1.plot(trial_data.Running_Average)
    ax1.plot(range(k),[avg for _ in range(k)])
    
    ax2.plot(trial_data.loc[k-100:,"Running_Average"])
    ax2.plot(range(k-100,k),[avg for _ in range(k-100,k)])
    plt.draw()

#observation: Compare the two plots. 
#1. The average reward approximately converges over a relatively small number 
#    of trials, but
#2. fine convergence takes a very long time, and
#3. there is a significant variation in the long term average, even for the same
#   seed. (I think this must be due to a subtle error)
