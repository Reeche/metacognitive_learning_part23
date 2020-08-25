import numpy as np

from heuristics import lex, equalweight

class experiment1():
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

        # next state = next p
        #next_p = np.random.uniform(0.9, 1)
        next_p = 0.6

        return outcome, execution_time, next_p

    def reward(self, selected_choice, execution_time):
        if selected_choice.endswith("1"): #selected gamble 1
            reward = np.random.choice([float(self.x1), float(self.y1)], 1, p=[self.p, 1 - self.p])
        else:
            reward = np.random.choice([float(self.x2), float(self.y2)], 1, p=[self.p, 1 - self.p])
        return reward - execution_time



# for _ in range(10):
#     # gamble 1
#     x_1 = np.random.uniform(-10, 10, size=1)
#     y_1 = np.random.uniform(-10, 10, size=1)
#
#     # gamble 2
#     x_2 = np.random.uniform(-10, 10, size=1)
#     y_2 = np.random.uniform(-10, 10, size=1)
#
#     #p = np.random.uniform(0.5, 1)
#     p = 0.5 # when p = 0.5, both methods output the same results
#     #p = 0.9 #when p = 0.9, equal weight choose the largest value, which might be unlikely
#
#
#     probability_dict = {"x1": p,
#                         "y1": 1 - p,
#                         "x2": p,
#                         "y2": 1 - p}
#
#     values_dict = {"x1": x_1,
#                    "y1": y_1,
#                    "x2": x_2,
#                    "y2": y_2}
#
#
#     print(values_dict)
#     print(probability_dict)
#     lex_outcome, used_time = lex(probability_dict, values_dict)
#     print("Using the LEX heuristics: ")
#     if lex_outcome.endswith("1"):
#         print("Gamble 1 is chosen!", values_dict[lex_outcome] - used_time, "with probability ", probability_dict[lex_outcome])
#     else:
#         print("Gamble 2 is chosen!", values_dict[lex_outcome] - used_time, "with probability ", probability_dict[lex_outcome])
#
#     equalweight_outcome, used_time = equalweight(values_dict)
#     print("Using the equal weight heuristics: ")
#     if equalweight_outcome.endswith("1"):
#         print("Gamble 1 is chosen!", values_dict[equalweight_outcome] - used_time, "with probability ", probability_dict[equalweight_outcome])
#     else:
#         print("Gamble 2 is chosen!", values_dict[equalweight_outcome] - used_time, "with probability ", probability_dict[equalweight_outcome])
#
#     print("------------------------")



