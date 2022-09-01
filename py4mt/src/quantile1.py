#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 17:20:13 2021
https://stats.stackexchange.com/questions/103258/online-estimation-of-quartiles-without-storing-observations

import numpy as np
import matplotlib.pyplot as plt

distribution = np.random.normal

running_percentile = RunningPercentile(0.841,step)
observations = []
for _ in range(1000000):
    observation = distribution()
    running_percentile.push(observation)
    observations.append(observation)

plt.figure(figsize=(10, 3))
plt.hist(observations, bins=100)
plt.axvline(running_percentile.x, c='k')
plt.show()

M = Q1 = Q3 = first data value
step =step_Q1 = step_Q3 = a small value
for each new data :
        # update median M
        if M > data:
            M = M - step
        elif M < data:
            M = M + step
        if abs(data-M) < step:
            step = step /2

        # estimate Q1 using M
        if data < M:
            if Q1 > data:
                Q1 = Q1 - step_Q1
            elif Q1 < data:
                Q1 = Q1 + step_Q1
            if abs(data - Q1) < step_Q1:
                step_Q1 = step_Q1/2
        # estimate Q3 using M
        elif data > M:
            if Q3 > data:
                Q3 = Q3 - step_Q3
            elif Q3 < data:
                Q3 = Q3 + step_Q3
            if abs(data-Q3) < step_Q3:
                step_Q3 = step_Q3 /2



"""

class RunningPercentile:
    def __init__(self, percentile=0.5, step=0.1):
        self.step = step
        self.step_up = 1.0 - percentile
        self.step_down = percentile
        self.x = None

    def push(self, observation):
        if self.x is None:
            self.x = observation
            return

        if self.x > observation:
            self.x -= self.step * self.step_up
        elif self.x < observation:
            self.x += self.step * self.step_down
        if abs(observation - self.x) < self.step:
            self.step /= 2.0
