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

# calculate the medians
print("Medians")
print(np.median(h_6))
print(np.median(h_9))
print(np.median(rl_6))
print(np.median(rl_9))

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

# plot the learning curve for human vs. rl with condition p = 0.6
plt.plot(h_6, label="human participant")
plt.plot(rl_6, label="reinforcement learning agent")
plt.legend()
plt.savefig("learning_curve_6.png")
plt.show()

# plot the learning curve for human vs. rl with condition p = 0.6
plt.plot(h_9, label="human participant")
plt.plot(rl_9, label="reinforcement learning agent")
plt.legend()
plt.savefig("learning_curve_9.png")


# t-test
print("T-test", stats.ttest_ind(h_6,h_9))

# Wilcoxon rank-sum test (less assumptions than t-test)
print("Wilcoxon test comparing two groups of participants", stats.ranksums(h_6, h_9))
print("Wilcoxon test comparing two groups of rl agent", stats.ranksums(rl_6, rl_9))
print("Wilcoxon test comparing participants with rl for p=0.6", stats.ranksums(h_6, rl_6))
print("Wilcoxon test comparing participants with rl for p=0.9", stats.ranksums(h_9, rl_9))

# credible intervals
print("Credible interval for h_6", stats.bayes_mvs(h_6, alpha=0.95))
print("Credible intervalfor h_9", stats.bayes_mvs(h_9, alpha=0.95))
print("Credible interval for rl_6", stats.bayes_mvs(rl_6, alpha=0.95))
print("Credible interval for rl_9", stats.bayes_mvs(rl_9, alpha=0.95))