import numpy as np

class LinearModel(object):

	def __init__(self, number_of_features, number_of_actions):
		self._T = np.zeros((number_of_actions, number_of_features, number_of_features))
		self._R = np.zeros((number_of_actions, number_of_features))
		self._G = np.zeros((number_of_actions, number_of_features))

	def next_state(self, s, a):
		return np.matmul(self._T[a], s)

	def reward(self, s, a):
		return np.matmul(self._R[a], s)

	def discount(self, s, a):
		return np.matmul(self._G[a], s)

	def transition(self, state, action):
		return (
			self.reward(state, action),
			self.discount(state, action),
			self.next_state(state, action))

	def update(self, state, action, reward, discount, next_state, step_size=0.1):
		last_reward, last_discount, last_state = self.transition(state, action)
		temp_state = np.reshape(next_state - last_state, [1, -1])
		self._T[action] += step_size * np.outer(temp_state, state)
		self._R[action] += step_size * (reward - last_reward) * state
		self._G[action] += step_size * (discount - last_discount) * state


class ExperienceQ():

	def __init__(
		self, number_of_features, number_of_actions, *args, **kwargs):
		super(ExperienceQ, self).__init__(
			number_of_actions=number_of_actions, *args, **kwargs)
		self._T = np.zeros((number_of_actions, number_of_features))

	# self._linear_model = LinearModel(number_of_features, number_of_actions)

	def q(self, state):
		return np.matmul(self._T, state)

	def step(self, reward, discount, next_state):
		s = self._state
		a = self._action
		r = reward
		g = discount
		next_s = next_state
		self._action = self._behaviour_policy(self._q)

		# Next line is to play around with epsilon greedy
		# self._action = self._behaviour_policy(self.q(next_s), 0.1)

		# append to replay buffer
		self._buffer.append([s, a, r, g, next_s])

		q_ = self.q(s)[a]
		q_next = self.q(next_s)
		self._T[a] += self._step_size * (r + g * np.max(q_next) - q_) * s

		# self._linear_model.update(s, a, r, g, next_s, self._step_size)

		if len(self._buffer) >= (self._num_offline_updates):
			for _ in range(self._num_offline_updates):
				i = np.random.choice(len(self._buffer))
				bs, ba, br, bg, bnext_s = self._buffer[i]

				bq = self.q(bs)[ba]
				bq_next = self.q(bnext_s)
				self._T[ba] += self._step_size * (br + bg * np.max(bq_next) - bq) * bs
			# self._T[ba] += self._step_size * (br + bg * np.max(self.q(bnext_s)) - self.q(bs)[ba]) * bs

		self._state = next_s
		return self._action


class DynaQ():

	def __init__(self, number_of_features, number_of_actions, *args, **kwargs):
		super(DynaQ, self).__init__(
			number_of_actions=number_of_actions, *args, **kwargs)
		self._linear_model = LinearModel(number_of_features, number_of_actions)
		self._T = np.zeros((number_of_actions, number_of_features))

	def q(self, state):
		return np.matmul(self._T, state)

	def step(self, reward, discount, next_state):
		s = self._state
		a = self._action
		r = reward
		g = discount
		next_s = next_state
		self._action = self._behaviour_policy(next_s)

		# Next line is to play around with epsilon greedy
		# self._action = self._behaviour_policy(self.q(next_s), 0.1)

		# append to buffer
		self._buffer.append([s, a, r, g, next_s])

		q = self.q(s)[a]
		q_next = self.q(next_s)
		self._T[a] += self._step_size * (r + g * np.max(q_next) - q) * s

		# update model
		self._linear_model.update(s, a, r, g, next_s, self._step_size)

		if len(self._buffer) > (self._num_offline_updates - 1):
			for _ in range(self._num_offline_updates):
				i = np.random.choice(len(self._buffer))
				bs, ba, br, bg, bnext_s = self._buffer[i]
				mr, mg, mnext_s = self._linear_model.transition(bs, ba)

				mq = np.matmul(self._T[ba], bs)
				mq_next = np.matmul(self._T, mnext_s)
				self._T[ba] += self._step_size * (mr + mg * np.max(mq_next) - mq) * bs

		self._state = next_s

		return self._action