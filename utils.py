import numpy as np

def discretizeP(p):
	return np.round(p, decimals=1)

def experiment1_transformation(p):
	p = np.round(p, decimals=1)
	if p == 0.6 or p == 0.5 or p == 0.4:
		return 0
	else:
		return 1

def experiment2_transformation(p):
	p = np.round(p, decimals=1)
	if p == 0.5:
		return 0
	else:
		return 1

def experiment3_transformation(p):
	p = np.round(p, decimals=1)
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


def get_env_parameters():
	x1 = np.random.uniform(-10, 10, size=1)
	y1 = np.random.uniform(-10, 10, size=1)

	# gamble 2
	x2 = np.random.uniform(-10, 10, size=1)
	y2 = np.random.uniform(-10, 10, size=1)

	# get p
	#p = np.random.uniform(0, 1)
	p = 0.5
	return x1, y1, x2, y2, p