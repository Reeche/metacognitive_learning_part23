import time

# one outcome is more likely
def lex(probability_dict, values_dict):
    """
    The Lexicographic heuristic (LEX). It first checks for the two more probable outcomes and then chooses the one with the
    higher value.

    :param probability_dict: a dictionary combining the payoffs and their corresponding probabilities
    :param values_dict: a dictionary combining the payoffs and their corresponding values
    :return: the outcome with the largest value according to the LEX rule (see description), execution time
    """
    start_time = time.time()

    # check value with highest probability and saves into a list
    prob_values = [k for k, v in probability_dict.items() if v == max(probability_dict.values())]

    # filter the values_dict using the selected values
    temp = {}
    for key in prob_values:
        # select the key item pair from the values_dict
        values = values_dict[key]
        temp.update({key: values})

    return max(temp, key=temp.get), 0.1

    # uncomment the following line if you want to use computational execution time instead of manual set execution time
    # return max(temp, key=temp.get), (time.time() - start_time)


# all outcomes almost equally likely
def equalweight(values_dict):
    """
    The Equal-Weight heuristic (EQW). It ignores the probabilities and only checks for the outcome with the largest value.
    If two or more outcomes have the same value, it will choose the first one in the list.

    :param values_dict: a dictionary combining the payoffs and their corresponding values
    :return: the outcome with the largest value according to the EQW rule (see description), execution time
    """
    start_time = time.time()

    # all values equally likely, therefore only choose the highest value
    return max(values_dict, key=values_dict.get), 0

    # uncomment the following line if you want to use computational execution time instead of manual set execution time
    # return max(values_dict, key=values_dict.get), (time.time() - start_time)
