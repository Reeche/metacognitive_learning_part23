import numpy as np
import matplotlib.pyplot as plt

from env import environment
from sarsa import SarsaAgent
from utils import experiment_transformation, get_env_parameters

episodes = 6
steps = 5000
experiment = 1

# initialise the SARSA agent
if experiment == 1 or experiment == 2:
	agent = SarsaAgent(2, 2)
else:
	agent = SarsaAgent(6, 2)
#agent = Qagent(2, 2)

# init state and env
x1, y1, x2, y2, p = get_env_parameters(experiment)

# initiate env
env = environment(x1, y1, x2, y2, p, experiment)

overall_reward_list = []
overall_strategy_list = []
for _ in range(10): #average over plots
	episode_strategy_list = []
	episode_reward_list = []
	for _ in range(episodes):
		step_reward_list = []
		step_strategy_list = []

		state = experiment_transformation(experiment, p)
		action = agent.act(state)

		for _ in range(steps):
			# take action and observe reward, next_state
			outcome, execution_time, next_p = env.step(action)
			reward = env.reward(outcome, execution_time)
			next_state = experiment_transformation(experiment, next_p)

			# choose next action based on new state
			next_action = agent.act(next_state)

			# update Q table
			agent.update(state, action, reward, next_state, next_action)

			state = next_state
			action = next_action

			step_reward_list.insert(len(step_reward_list), reward)
			step_strategy_list.insert(len(step_strategy_list), action)

			# get new set of env parameters without p because next p is already chosen
			x1, y1, x2, y2, _ = get_env_parameters(experiment)

			# initiate env
			env = environment(x1, y1, x2, y2, next_p, experiment)

		episode_reward_list.insert(len(episode_reward_list), sum(step_reward_list))
		episode_strategy_list.insert(len(episode_strategy_list), sum(step_strategy_list)) #outputs sum of one episode

	# outputs average over episodes
	overall_reward_list.insert(len(overall_reward_list), np.array(episode_reward_list) / steps)
	overall_strategy_list.insert(len(overall_strategy_list), np.array(episode_strategy_list)/steps)

# calculate the average across runs
avg_overall_reward_list = np.sum(np.array(overall_reward_list), axis=0)/10
avg_overall_strategy_list = np.sum(np.array(overall_strategy_list), axis=0)/10

# calculate the difference between runs in order to plot the spread
action_std = np.std(np.array(overall_strategy_list), axis=0)

print("values", avg_overall_strategy_list)
plt.figure(1)
plt.subplot(211)
plt.errorbar(x=range(6), y=avg_overall_strategy_list, yerr=action_std, label="Averaged proportion of EQW over 10 runs")
plt.legend()
plt.subplot(212)
plt.plot(avg_overall_reward_list, label="Averaged reward over 10 runs")
plt.legend()
plt.show()



