import numpy as npimport randomdef experiment_transformation(experiment, p):	"""	Select state based on the experiment and p. For experiment 1, there are two state "equal" and "not equal".	For experiment 2 and 4 there is only one state since p is fixed to one value.	For experiment 3, there are 6 states because I have rounded p to one decimal place.	:param experiment: the selected experiment {1, 2, 3, 4}	:param p: the probability parameter p	:return: the state	"""	if experiment == 1:		if p <= 0.6:			return 0		else:			return 1	if experiment == 2:		return 0	if experiment == 3:		p = np.round(p, decimals=1)		if p == 0.5:			return 0		if p == 0.6:			return 1		if p == 0.7:			return 2		if p == 0.8:			return 3		if p == 0.9:			return 4		if p == 1:			return 5		else:			raise ValueError("Invalid p for experiment 3 given!", p)	if experiment == 4:		return 0def get_env_parameters(experiment):	"""	Get the parameters x1, y1, x2, y2, which all follow uniform[-10, 10] as well as p	random.uniform INCLUDES both lower and upper values	:param experiment: the selected experiment	:return: parameters x1, y1, x2, y2 and p	"""	# gamble 1	x1 = random.uniform(-10, 10)	y1 = random.uniform(-10, 10)	# gamble 2	x2 = random.uniform(-10, 10)	y2 = random.uniform(-10, 10)	# get p	p = select_p_based_on_experiment(experiment)	return x1, y1, x2, y2, pdef select_p_based_on_experiment(experiment):	"""	Sample p based on the experiment. If experiment 1 or 3, p is sampled from uniform[0.5, 1].	If experiment is 2, p is sampled either from uniform[0.5, 0.6) or uniform(0.9, 1]	If experiment is 4, p is set to either 0.6 or 0.9	numpy.random.uniform INCLUDES lower value but EXCLUDES upper value	:param experiment: the selected experiment	:return: value for the probability parameter p	"""	if experiment == 1 or experiment == 3:		return random.uniform(0.5, 1)	if experiment == 2:		#return np.random.uniform(0.5, 0.6) #EXLUDES 0.6		return random.uniform(0.9+1e-16, 1) #1+e-16 was chosen as it is the default float for python	if experiment == 4:		# for comparison with human participants		return 0.6		#return 0.9