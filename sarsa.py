import numpy as np


class SarsaAgent():

    def __init__(self, num_state, num_action, gamma=0.99, epsilon=0.15, lr=0.1, epsilon_decay=0.9999):
        """
        The SARSA agent is initiated with the following parameters:

        :param num_state: number of states
        :param num_action: number of actions
        :param gamma: gamma
        :param epsilon: exploration rate
        :param lr: learning rate
        :param epsilon_decay: epsilon decay rate
        """
        self.num_state = num_state
        self.num_action = num_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.reset()

    def reset(self):
        """
        Resets the Q-table to all 0
        """
        self.q = [[0 for j in range(self.num_action)] for i in range(self.num_state)]

    # get action
    def act(self, state):
        """
        Choose an action based on epsilon greedy method. With a random action is chosen with the probability epsilon.

        :param state: the current state
        :return:
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_action)
            return action
        else:
            action = np.argmax(self.q[state])
            return action


    def update(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-table

        :param state: the current state
        :param action: the currently chosen action
        :param reward: the reward
        :param next_state: next state
        :param next_action: next action
        :return: An updated Q-table and new epsilon
        """
        diff_q = reward + self.gamma * self.q[next_state][next_action] - self.q[state][action]
        self.q[state][action] = self.q[state][action] + self.lr * diff_q

        self.epsilon *= self.epsilon_decay
