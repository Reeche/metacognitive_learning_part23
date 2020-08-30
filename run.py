import numpy as np
import matplotlib.pyplot as plt

from env import environment
from sarsa import SarsaAgent
from utils import experiment_transformation, get_env_parameters

# enter the number of episodes and steps you want to run
episodes = 6
steps = 5000

# Experiment number {1, 2, 3, 4}, experiment 4 is for the part III for p = 0.6
experiment = 4

# initialise the SARSA agent, number of states depend on the experiment
if experiment == 1:
	agent = SarsaAgent(2, 2)
elif experiment == 3:
	agent = SarsaAgent(6, 2)
elif experiment == 2 or experiment == 4:
	agent = SarsaAgent(1, 2)


# sample the parameters to pass on to the environment
x1, y1, x2, y2, p = get_env_parameters(experiment)

# initiate environment given the parameters
env = environment(x1, y1, x2, y2, p, experiment)

overall_reward_list = []
overall_strategy_list = []

# number of runs to average over
for _ in range(10):
	episode_strategy_list = []
	episode_reward_list = []

	for _ in range(episodes):
		step_reward_list = []
		step_strategy_list = []

		# determine which state you are in depending on p
		state = experiment_transformation(experiment, p)

		# select the action (heuristic) based on the state
		action = agent.act(state)

		for _ in range(steps):
			# take action (heuristic) and observe outcome, execution time and next p
			outcome, execution_time, next_p = env.step(action)

			# reward is calculated based in the selected outcome
			reward = env.reward(outcome, execution_time)

			# the next state the determined based on p
			next_state = experiment_transformation(experiment, next_p)

			# choose next action based on next state
			next_action = agent.act(next_state)

			# update Q table
			agent.update(state, action, reward, next_state, next_action)

			# update the state and selected action (heuristic)
			state = next_state
			action = next_action

			# insert reward and action into a list
			step_reward_list.insert(len(step_reward_list), reward)
			step_strategy_list.insert(len(step_strategy_list), action)

			# get new set of env parameters without p because next p is already chosen
			x1, y1, x2, y2, _ = get_env_parameters(experiment)

			# initiate environment again with new values for next trial
			env = environment(x1, y1, x2, y2, next_p, experiment)

		# these lists record the sums over one episode
		episode_reward_list.insert(len(episode_reward_list), sum(step_reward_list))
		episode_strategy_list.insert(len(episode_strategy_list), sum(step_strategy_list)) #

	# these lists output  the averages over episodes
	overall_reward_list.insert(len(overall_reward_list), np.array(episode_reward_list) / steps)
	overall_strategy_list.insert(len(overall_strategy_list), np.array(episode_strategy_list)/steps)

# calculate the average across runs
avg_overall_reward_list = np.sum(np.array(overall_reward_list), axis=0)/10
avg_overall_strategy_list = np.sum(np.array(overall_strategy_list), axis=0)/10

# calculate the standard deviation between runs
action_std = np.std(np.array(overall_strategy_list), axis=0)

print("values", avg_overall_strategy_list)

# plot the results
# plt.figure(1)
# plt.subplot(211)
plt.errorbar(x=range(6), y=avg_overall_strategy_list, yerr=action_std, label="Averaged proportion of EQW over 10 runs")
plt.ylabel("Frequency")
plt.xlabel("Block number")
plt.legend()
# plt.subplot(212)
# plt.plot(avg_overall_reward_list, label="Averaged reward over 10 runs")
# plt.legend()
plt.show()



