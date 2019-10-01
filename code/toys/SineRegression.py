"""
The Sine Regression toy problem as first described by Finn et al.
(2017). To quote from the paper:

 Each task involves regressing from the input to the output of a 
 sine wave, where the amplitude and phase of the sinusoid are 
 varied between tasks. Thus, p(T) is continuous, where the 
 amplitude varies within [0.1, 5.0] and the phase varies within 
 [0, π], and the input and output both have a dimensionality of 
 1. During training and testing, datapoints x are sampled 
 uniformly from [−5.0, 5.0]. The loss is the mean-squared error 
 between the prediction f(x) and true value.

Note here that p(T) refers to the distribution over tasks. 
"""

import numpy as np
import matplotlib.pyplot as plt


class SineTask:

	def __init__(self, ampl, phase, x_range=[-5, 5]):
		self.ampl = ampl
		self.phase = phase
		self.x_range = x_range

	def generate_points(self, n=10):
		x = self.x_range[0] + np.random.rand(n, 1) * (self.x_range[1] - self.x_range[0])
		y = self.ampl * np.sin(x + self.phase)
		return x, y

	def plot(self):
		fig, ax = plt.subplots()
		plot_x = np.arange(-5, 5, 0.1)
		plot_y = self.ampl * np.sin(plot_x + self.phase)
		ax.plot(plot_x, plot_y)
		points = self.generate_points()
		ax.scatter(points[0], points[1])
		plt.show()

class SineRegressionDist:

	def __init__(self, ampl_range=[0.1, 5], phase_range=[0, np.pi]):
		self.ampl_range = ampl_range
		self.phase_range = phase_range

	def sampler(self, rnge):
		return rnge[0] + np.random.rand() * (rnge[1] - rnge[0])

	def generate_task(self):
		return SineTask(self.sampler(self.ampl_range), self.sampler(self.phase_range))


if __name__ == "__main__":
	dist = SineRegressionDist()
	task = dist.generate_task()
	print(task.generate_points())
	task.plot()
