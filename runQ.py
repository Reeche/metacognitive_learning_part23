import numpy as np
import matplotlib.pyplot as plt

from env import experiment1
from TabularQ import TableQAgent


agent = TableQAgent()

def discretizeP_exp1(p):
	#classify p as equal or not equal
	p = np.round(p, decimals=1)
	if p == 0.6 or p == 0.5 or p == 0.4:
		return 0
	else:
		return 1

def discretizeP(p):
	return np.round(p, decimals=1)

#p = np.random.uniform(0.9, 1)
p = 0.6
episode = 10
steps = 2000

sum_strategy_list = []
sum_reward_list = []
for _ in range(episode): #equivalent to human experiment 6 episodes
	reward_list = []
	strategy_list = []
	x1 = np.random.uniform(-10, 10, size=1)
	y1 = np.random.uniform(-10, 10, size=1)

	# gamble 2
	x2 = np.random.uniform(-10, 10, size=1)
	y2 = np.random.uniform(-10, 10, size=1)

	# get p
	# = np.random.uniform(0, 1)

	# initiate env
	env = experiment1(x1, y1, x2, y2, p)

	# classify p as equal or not equal
	# state = discretizeP_exp1(p)
	state = int(discretizeP(p) * 10)

	strategy = agent.act(state)

	for _ in range(steps): # with 10 steps each
		# gamble 1
		x1 = np.random.uniform(-10, 10, size=1)
		y1 = np.random.uniform(-10, 10, size=1)

		# gamble 2
		x2 = np.random.uniform(-10, 10, size=1)
		y2 = np.random.uniform(-10, 10, size=1)

		# get p
		# = np.random.uniform(0, 1)

		# initiate env
		env = experiment1(x1, y1, x2, y2, p)

		#classify p as equal or not equal
		#state = discretizeP_exp1(p)
		state = int(discretizeP(p) * 10)

		# choose action according to p
		#strategy = agent.act(state) #returns strategy LEX or EQW

		#print(action)

		# observe reward according to chosen action
		outcome, execution_time, next_p = env.step(strategy) #next_state is the next
		reward = env.reward(outcome, execution_time)
		#new_state = discretizeP_exp1(next_p)
		new_state = int(discretizeP(next_p) * 10)

		next_strategy = agent.act(new_state) # action based on the new state

		# update Q table
		agent.update(state, strategy, reward, new_state, next_strategy)

		#state = new_state
		p = next_p
		strategy = next_strategy

		reward_list.insert(len(reward_list), reward)
		strategy_list.insert(len(strategy_list), strategy)

	#print(reward_list)
	#print("SUM OF ACTIONS", sum(action_list))
	sum_reward_list.insert(len(sum_reward_list), sum(reward_list))
	sum_strategy_list.insert(len(sum_strategy_list), sum(strategy_list))
#print(sum_action_list)

len = episode * steps
plt.figure(1)
plt.subplot(211)
plt.plot(np.array(sum_strategy_list)/steps, label="Proportion of EQW")
plt.legend()
plt.subplot(212)
plt.plot(np.array(sum_reward_list)/steps, label="Accumulated reward") #todo: rename because it is not
plt.legend()
plt.show()



