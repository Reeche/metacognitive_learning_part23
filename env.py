import numpy as np

from heuristics import lex, equalweight
from utils import select_p_based_on_experiment

class environment():
    def __init__(self, x1, y1, x2, y2, p):
        # gamble 1
        self.x1 = x1
        self.y1 = y1

        # gamble 2
        self.x2 = x2
        self.y2 = y2

        self.p = p

    def step(self, strategy):
        """

        :param action: the selected action. Either LEX (denoted by 0) or EQW (denoted by 1)
        :return: the selected outcome as str, the reward
        """
        probability_dict = {"x1": self.p,
                            "y1": 1 - self.p,
                            "x2": self.p,
                            "y2": 1 - self.p}

        values_dict = {"x1": self.x1,
                       "y1": self.y1,
                       "x2": self.x2,
                       "y2": self.y2}

        #print(probability_dict)
        #print(values_dict)
        if strategy == 0: #LEX
            outcome, execution_time = lex(probability_dict, values_dict)
            # = values_dict[outcome] - execution_time
            #print("LEX reward", reward, outcome, execution_time)
        else: #EQW
            outcome, execution_time = equalweight(values_dict)
            #reward = values_dict[outcome] - execution_time
            #print("EQW reward", reward, outcome, execution_time)

        # get p
        next_p = select_p_based_on_experiment(3)

        return outcome, execution_time, next_p

    def reward(self, selected_choice, execution_time):
        if selected_choice.endswith("1"): #selected gamble 1
            reward = np.random.choice([float(self.x1), float(self.y1)], 1, p=[self.p, 1 - self.p])
        else:
            reward = np.random.choice([float(self.x2), float(self.y2)], 1, p=[self.p, 1 - self.p])
        return reward - execution_time

