"""Upper Confidence Bound (UCB) for Reinforcement Learning."""

# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("dataset.csv")

# Implementing UCB
N, d, selected_ads = 10000, 10, []

numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
total_rewards = 0

for n in range(N):

    ad = 0
    max_upper_bound = 0

    for i in range(0, d):

        if (numbers_of_selections[i] > 0):

            average_reward = sum_of_rewards[i] / numbers_of_selections[i]
            delta_i = np.sqrt(3/2 * np.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i

        else:

            upper_bound = 1e400

        if upper_bound > max_upper_bound:

            max_upper_bound = upper_bound
            ad = i

    selected_ads.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_rewards += reward

# Visualizations
plt.hist(selected_ads)

plt.title("Histogram")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()
