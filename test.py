import numpy as np

from env import environment
from heuristics import lex, equalweight

for _ in range(10):
    # gamble 1
    x_1 = np.random.uniform(-10, 10, size=1)
    y_1 = np.random.uniform(-10, 10, size=1)

    # gamble 2
    x_2 = np.random.uniform(-10, 10, size=1)
    y_2 = np.random.uniform(-10, 10, size=1)

    #p = np.random.uniform(0.5, 1)
    p = 0.5 # when p = 0.5, both methods output the same results
    #p = 0.9 #when p = 0.9, equal weight choose the largest value, which might be unlikely


    probability_dict = {"x1": p,
                        "y1": 1 - p,
                        "x2": p,
                        "y2": 1 - p}

    values_dict = {"x1": x_1,
                   "y1": y_1,
                   "x2": x_2,
                   "y2": y_2}


    print(values_dict)
    print(probability_dict)
    lex_outcome, used_time = lex(probability_dict, values_dict)
    print("Using the LEX heuristics: ")
    if lex_outcome.endswith("1"):
        print("Gamble 1 is chosen!", values_dict[lex_outcome] - used_time, "with probability ", probability_dict[lex_outcome])
    else:
        print("Gamble 2 is chosen!", values_dict[lex_outcome] - used_time, "with probability ", probability_dict[lex_outcome])

    equalweight_outcome, used_time = equalweight(values_dict)
    print("Using the equal weight heuristics: ")
    if equalweight_outcome.endswith("1"):
        print("Gamble 1 is chosen!", values_dict[equalweight_outcome] - used_time, "with probability ", probability_dict[equalweight_outcome])
    else:
        print("Gamble 2 is chosen!", values_dict[equalweight_outcome] - used_time, "with probability ", probability_dict[equalweight_outcome])

    print("------------------------")


