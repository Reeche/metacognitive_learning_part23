import numpy as np
import matplotlib.pyplot as plt
import statistics

from env import environment
from sarsa import SarsaAgent
from utils import experiment_transformation, get_env_parameters

episodes = 6
steps = 5000
experiment = 2

# initialise the SARSA agent
agent = SarsaAgent(2, 2)
#agent = Qagent(2, 2)

# init state and env
x1, y1, x2, y2, p = get_env_parameters(experiment)

# initiate env
env = environment(x1, y1, x2, y2, p, experiment)

action_overall_average_list = []
reward_overall_average_list = []
for _ in range(5): #average over plots
	sum_strategy_list = []
	sum_reward_list = []
	for _ in range(episodes):
		reward_list = []
		strategy_list = []

		state = experiment_transformation(experiment, p) #p should be uniform float
		action = agent.act(state)

		for _ in range(steps):
			# take action and observe reward, next_state
			#print(action) #should 0 or 1
			outcome, execution_time, next_p = env.step(action)
			reward = env.reward(outcome, execution_time)

			# choose next state
			next_state = experiment_transformation(experiment, next_p)

			# choose next action based on new state
			next_action = agent.act(next_state)

			# update Q table
			agent.update(state, action, reward, next_state, next_action)

			state = next_state
			action = next_action

			reward_list.insert(len(reward_list), reward)
			strategy_list.insert(len(strategy_list), action)

			# init state and env
			x1, y1, x2, y2, _ = get_env_parameters(experiment)

			# initiate env
			env = environment(x1, y1, x2, y2, next_p, experiment)


		sum_reward_list.insert(len(sum_reward_list), sum(reward_list))
		sum_strategy_list.insert(len(sum_strategy_list), sum(strategy_list)) #outputs sum of one episode

	# outputs average over episodes
	action_overall_average_list.insert(len(action_overall_average_list), np.array(sum_strategy_list)/steps)
	reward_overall_average_list.insert(len(reward_overall_average_list), np.array(sum_reward_list)/steps)

# calculate the average across runs
action_values_across_runs = np.sum(np.array(action_overall_average_list), axis=0)
reward_overall_average_list = np.sum(np.array(reward_overall_average_list), axis=0)

# calculate the difference between runs in order to plot the spreaf
action_std = np.std(np.array(action_overall_average_list), axis=0)

print("values", action_values_across_runs/5)
plt.figure(1)
plt.subplot(211)
plt.errorbar(x=range(6), y=action_values_across_runs/5, yerr=action_std, label="Averaged proportion of EQW over 5 runs")
plt.legend()
plt.subplot(212)
plt.plot(np.array(reward_overall_average_list)/5, label="Averaged reward over 5 runs")
plt.legend()
plt.show()



