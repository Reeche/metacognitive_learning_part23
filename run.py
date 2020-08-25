import numpy as np
import matplotlib.pyplot as plt

from env import experiment1
from sarsa import SarsaAgent
from qlearning import Qagent
from utils import experiment1_transformation, get_env_parameters

episodes = 20
steps = 2000

sum_strategy_list = []
sum_reward_list = []

# initialise the SARSA agent
agent = SarsaAgent(2, 2)
#agent = Qagent(2, 2)

# init state and env
x1, y1, x2, y2, p = get_env_parameters()

# initiate env
env = experiment1(x1, y1, x2, y2, p)

for _ in range(episodes):
	reward_list = []
	strategy_list = []

	state = experiment1_transformation(p)
	action = agent.act(state)

	for _ in range(steps):
		# take action and observe reward, next_state
		outcome, execution_time, next_p = env.step(action)
		reward = env.reward(outcome, execution_time)
		next_state = experiment1_transformation(next_p)

		# choose next action based on new state
		next_action = agent.act(next_state)

		# update Q table
		agent.update(state, action, reward, next_state, next_action) #SARSA

		state = next_state
		action = next_action

		reward_list.insert(len(reward_list), reward)
		strategy_list.insert(len(strategy_list), action)

		# init state and env
		x1, y1, x2, y2, _ = get_env_parameters()

		# initiate env
		env = experiment1(x1, y1, x2, y2, next_p)


	sum_reward_list.insert(len(sum_reward_list), sum(reward_list))
	sum_strategy_list.insert(len(sum_strategy_list), sum(strategy_list))

plt.figure(1)
plt.subplot(211)
plt.plot(np.array(sum_strategy_list)/steps, label="Proportion of EQW")
plt.legend()
plt.subplot(212)
plt.plot(np.array(sum_reward_list)/steps, label="Accumulated reward") #todo: rename because it is not
plt.legend()
plt.show()
