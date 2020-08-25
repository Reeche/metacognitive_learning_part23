import time

# one outcome is more likely
def lex(probability_dict, values_dict):
    start_time = time.time()
    # check value with highest probability and saves into a list
    prob_values = [k for k, v in probability_dict.items() if v == max(probability_dict.values())]

    # filter the values_dict using the selected values
    temp = {}
    for key in prob_values:
        # select the key item pair from the values_dict
        values = values_dict[key]
        temp.update({key: values})
    #("Execution time of LEX algoritm", (time.time() - start_time))
    return max(temp, key=temp.get), (time.time() - start_time) #todo: check faster implementations

# all outcomes almost equally likely
def equalweight(values_dict):
    start_time = time.time()
    # all values equally likely, therefore only choose the highest value
    #print("Execution time of EQW algorithm", (time.time() - start_time))
    return max(values_dict, key=values_dict.get), (time.time() - start_time)