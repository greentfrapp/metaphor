"""
Model-Agnostic Meta-Learning
Finn et al.
https://arxiv.org/abs/1703.03400
"""

"""
Notes to Self

Treat a single sample as
	X features: x1,y1,x2,y2,...,xn-1,yn-1,xn
	Y label: yn
A "single pass" through the architecture should produce
a prediction that we can use to calculate loss against
the label
"""

import tensorflow as tf

class MAML:

	def __init__(self, input_dim=1, output_dim=1):

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden = [40, 40]

		self.metatrain_tr_x = tf.placeholder(
			shape=(None, self.input_dim),
			dtype=tf.float32,
			name='metatrain_tr_x',
		)
		self.metatrain_tr_y = tf.placeholder(
			shape=(None, self.output_dim),
			dtype=tf.float32,
			name='metatrain_tr_y',
		)
		self.metatrain_te_x = tf.placeholder(
			shape=(None, self.input_dim),
			dtype=tf.float32,
			name='metatrain_te_x',
		)
		self.metatrain_te_y = tf.placeholder(
			shape=(None, self.output_dim),
			dtype=tf.float32,
			name='metatrain_te_y',
		)

		self.params = {}
		for i, units in enumerate(self.hidden):
			if i == 0:
				weights = tf.Variable(
					initial_value=tf.truncated_normal([self.input_dim, units], stddev=0.01),
					trainable=True,
					name='weights_{}'.format(i),
				)
				bias = tf.Variable(
					initial_value=tf.zeros([units]),
					trainable=True,
					name='bias_{}'.format(i),
				)
			else:
				weights = tf.Variable(
					initial_value=tf.truncated_normal([self.hidden[i-1], units], stddev=0.01),
					trainable=True,
					name='weights_{}'.format(i),
				)
				bias = tf.Variable(
					initial_value=tf.zeros([units]),
					trainable=True,
					name='bias_{}'.format(i),
				)

			self.params['w{}'.format(i)] = weights
			self.params['b{}'.format(i)] = bias

		self.params['w{}'.format(len(hidden))] = tf.Variable(
			initial_value=tf.truncated_normal([self.hidden[len(self.hidden)-1], self.output_dim], stddev=0.01),
			trainable=True,
			name='weights_{}'.format(len(hidden)),
		)

		self.params['b{}'.format(len(hidden))] = tf.Variable(
			initial_value=tf.zeros([self.output_dim]),
			trainable=True,
			name='bias_{}'.format(len(hidden)),
		)

		self.learning_rate = 1e-2

		self.input_dim = input_dim
		self.hidden = [40, 40]
		self.activations = [tf.nn.relu, tf.nn.relu]

		# Get original curve
		output = self.metatrain_te_x
		for layer in range(3):
			output = tf.matmul(output, self.params['w{}'.format(layer)]) + self.params['b{}'.format(layer)]
			if layer != 2:
				output = tf.nn.relu(output)
		self.orig_output = output
		#END Get original curve
		
		output = self.metatrain_tr_x
		for layer in range(3):
			output = tf.matmul(output, self.params['w{}'.format(layer)]) + self.params['b{}'.format(layer)]
			if layer != 2:
				output = tf.nn.relu(output)
		self.output = output

		

		# for i, units in enumerate(self.hidden):
		# 	output = tf.layers.dense(
		# 		inputs=output,
		# 		units=units,
		# 		activation=self.activations[i],
		# 		kernel_initializer=tf.random_normal_initializer(),
		# 		name='dense_{}'.format(i),
		# 	)

		# self.output = tf.layers.dense(
		# 	inputs=output,
		# 	units=1, # assuming output_dim = 1
		# 	activation=tf.tanh,
		# 	kernel_initializer=tf.random_normal_initializer(),
		# 	name='output',
		# )

		# def single_pass(self):
		"""
		input x1y1, x2y2, x3
		output y3
		"""
		# First do one update iteration
		# set input to x1, x2
		# perform manual update against y1, y2

		# First update iteration
		training_loss = tf.reduce_mean((self.output - self.metatrain_tr_y) ** 2)
		gradients = tf.gradients(training_loss, list(self.params.values()))
		gradients = dict(zip(self.params.keys(), gradients))
		new_weights = {
			'w0': self.params['w0'] - self.learning_rate * gradients['w0'],
			'b0': self.params['b0'] - self.learning_rate * gradients['b0'],
			'w1': self.params['w1'] - self.learning_rate * gradients['w1'],
			'b1': self.params['b1'] - self.learning_rate * gradients['b1'],
			'w2': self.params['w2'] - self.learning_rate * gradients['w2'],
			'b2': self.params['b2'] - self.learning_rate * gradients['b2'],
		}

		# Second update iteration etc.
		n_updates = 2
		for update in range(n_updates - 1):
			output = self.metatrain_tr_x
			for layer in range(3):
				output = tf.matmul(output, new_weights['w{}'.format(layer)]) + new_weights['b{}'.format(layer)]
				if layer != 2:
					output = tf.nn.relu(output)
			training_loss = tf.reduce_mean((output - self.metatrain_tr_y) ** 2)
			gradients = tf.gradients(training_loss, list(self.params.values()))
			gradients = dict(zip(self.params.keys(), gradients))
			new_weights = {
				'w0': new_weights['w0'] - self.learning_rate * gradients['w0'],
				'b0': new_weights['b0'] - self.learning_rate * gradients['b0'],
				'w1': new_weights['w1'] - self.learning_rate * gradients['w1'],
				'b1': new_weights['b1'] - self.learning_rate * gradients['b1'],
				'w2': new_weights['w2'] - self.learning_rate * gradients['w2'],
				'b2': new_weights['b2'] - self.learning_rate * gradients['b2'],
			}
		self.output_n = output

		# Backprop to original weights
		# using metatrain_te
		output = self.metatrain_te_x
		for layer in range(3):
			output = tf.matmul(output, new_weights['w{}'.format(layer)]) + new_weights['b{}'.format(layer)]
			if layer != 2:
				output = tf.nn.relu(output)
		self.final_prediction = output
		self.final_loss = tf.reduce_mean((self.final_prediction - self.metatrain_te_y) ** 2)

		self.meta_optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.final_loss)




















