import numpy as np

def discretizeP(p):
	return np.round(p, decimals=1)

def experiment_transformation(n, p):
	p = np.round(p, decimals=1)
	if n == 1:
		if p == 0.6 or p == 0.5:
			return 0
		else:
			return 1
	if n == 2:
		if p == 0.5 or p == 0.6 or p == 0.9 or p == 1:
			return 0
		else:
			return 1
	if n == 3:
		if p == 0.5:
			return 0
		if p == 0.6:
			return 1
		if p == 0.7:
			return 2
		if p == 0.8:
			return 3
		if p == 0.9:
			return 4
		if p == 1:
			return 5
		else:
			raise ValueError("Invalid p for experiment 3 given!", p)


def get_env_parameters(experiment):
	x1 = np.random.uniform(-10, 10, size=1)
	y1 = np.random.uniform(-10, 10, size=1)

	# gamble 2
	x2 = np.random.uniform(-10, 10, size=1)
	y2 = np.random.uniform(-10, 10, size=1)

	# get p
	p = select_p_based_on_experiment(experiment)
	return x1, y1, x2, y2, p

def select_p_based_on_experiment(n):
	if n == 1 or n == 3:
		return np.random.uniform(0.5, 1)
	if n == 2:
		return np.random.uniform(0.5, 0.6)
		#return np.random.uniform(0.9, 1)