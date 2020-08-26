import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

h_6 = [0.4, 0.55, 0.65, 0.7, 0.75, 0.8]
h_9 = [0.65, 0.5, 0.45, 0.4, 0.35, 0.3]

rl_6 = [0.90448, 0.95202, 0.94812, 0.94028, 0.94664, 0.93904]
rl_9 = [0.01874, 0.01394, 0.01142, 0.01044, 0.00934, 0.00744]

# calculate the means
print("Means")
print(np.mean(h_6))
print(np.mean(h_9))
print(np.mean(rl_6))
print(np.mean(rl_9))

# calculate the standard deviations
print("Standard deviations")
print(np.std(h_6))
print(np.std(h_9))
print(np.std(rl_6))
print(np.std(rl_9))

# plot the learning curve for human participants
plt.plot(h_6, label="p=0.6")
plt.plot(h_9, label="p=0.9")
plt.legend()
plt.savefig("learning_curve_human.png")
plt.show()

# plot the learning curve for reinforcement learning agent
plt.plot(rl_6, label="p=0.6")
plt.plot(rl_9, label="p=0.9")
plt.legend()
plt.savefig("learning_curve_rl.png")

# t-test
print("T-test", stats.ttest_ind(h_6,h_9))