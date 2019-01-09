# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 14:53:56 2018

@author: carto
"""

import numpy as np
import pandas as pd
import math

#import dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#UCB
N = 10000
d = 10
ads = []
selections = [0] * d
rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_ub = 0
    for i in range(0, d):
        if (selections[i] > 0):
            average_reward = rewards[i] / selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / selections[i])
            UCB = average_reward + delta_i
        else:
            UCB = 1e400
        if UCB > max_ub:
            max_ub = UCB
            ad = i
    ads.append(ad)
    selections[ad] = selections[ad] + 1
    reward = data.values[n,ad]
    rewards[ad] = rewards[ad] + reward
    total_reward = total_reward + reward