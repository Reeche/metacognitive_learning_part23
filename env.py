import numpy as np

from heuristics import lex, equalweight
from utils import select_p_based_on_experiment

class environment():
    def __init__(self, x1, y1, x2, y2, p, experiment):
        """
        The environment is initiated with different parameters of x1, x2, y1, y2 and p.

        :param x1: uniform[-10, 10]
        :param y1: uniform[-10, 10]
        :param x2: uniform[-10, 10]
        :param y2: uniform[-10, 10]
        :param p: uniform[0.5, 1]
        :param experiment: which experiment is currently run, 1 denotes part II.a,
        2 denotes part II.b, 3 denotes part II.c, 4 denotes the part III
        """
        # gamble 1
        self.x1 = x1
        self.y1 = y1

        # gamble 2
        self.x2 = x2
        self.y2 = y2

        self.p = p
        self.experiment = experiment

    def step(self, strategy: int):
        """
        The environment receives the selected strategy/heuristic and outputs the selected outcome (x1, y1, x2 or y2).
        It also selects the next p.

        :param strategy: the selected heuristic/strategy (LEX or EQW) by the RL agent
        :return: the selected outcome (x1, y1, x2, y2), the execution time of the selected strategy and next p
        """
        # creates a dictionary combining the payoffs and their corresponding probabilities
        probability_dict = {"x1": self.p,
                            "y1": 1 - self.p,
                            "x2": self.p,
                            "y2": 1 - self.p}

        # creates a dictionary combining the payoffs and their corresponding values
        values_dict = {"x1": self.x1,
                       "y1": self.y1,
                       "x2": self.x2,
                       "y2": self.y2}

        if strategy == 0: #LEX
            outcome, execution_time = lex(probability_dict, values_dict)
        else: #EQW
            outcome, execution_time = equalweight(values_dict)

        # get p
        next_p = select_p_based_on_experiment(self.experiment)

        return outcome, execution_time, next_p

    def reward(self, selected_outcome: str, execution_time: float):
        """
        The selected outcome determines whether gamble 1 or gamble 2 is played
        (if x1 or y1 is chosen -> gamble 1; if x2 or y2 is chosen -> gamble 2).
        Based on the gamble, the reward is calculated based on the uniformly generated values and the probabilities.

        :param selected_outcome: the selected outcome (x1, y1, x2, y2)
        :param execution_time: the execution time of the used strategy
        :return:
        """
        if selected_outcome.endswith("1"): #selected gamble 1
            reward = np.random.choice([float(self.x1), float(self.y1)], 1, p=[self.p, 1 - self.p])
        else: #selected gamble 2
            reward = np.random.choice([float(self.x2), float(self.y2)], 1, p=[self.p, 1 - self.p])
        return reward - execution_time

