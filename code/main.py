from algos import MAML
from toys import SineRegressionDist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

task_dist = SineRegressionDist()

maml = MAML()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(int(1e5)):

	task = task_dist.generate_task()
	x, y = task.generate_points()
	tx, ty = task.generate_points(5)

	feed_dict = {
		maml.metatrain_tr_x: x,
		maml.metatrain_tr_y: y,
		maml.metatrain_te_x: tx,
		maml.metatrain_te_y: ty,
	}

	y1, yn, l, _ = sess.run([maml.output, maml.output_n, maml.final_loss, maml.meta_optimizer], feed_dict=feed_dict)

	if i % 100 == 0: print("Step {}: {}".format(i, l))

# fig, ax = plt.subplots()
# plt.scatter(x, y, color='black')
# plt.scatter(x, y1, color='red')
# plt.scatter(x, yn, color='blue')
# plt.show()
# quit()

task = task_dist.generate_task()
x, y = task.generate_points()
tx, ty = task.generate_points(5)

x = np.arange(-5,-3, 0.3)
x = x.reshape(-1, 1)
y = task.ampl * np.sin(x + task.phase)

tx = np.arange(-5, 5, 0.1)
tx = tx.reshape(-1, 1)
ty = task.ampl * np.sin(tx + task.phase)

feed_dict = {
	maml.metatrain_tr_x: x,
	maml.metatrain_tr_y: y,
	maml.metatrain_te_x: tx
}

original_y, predicted_y = sess.run([maml.orig_output, maml.final_prediction], feed_dict=feed_dict)

fig, ax = plt.subplots()
ax.plot(tx, original_y, color="black")
ax.plot(tx, ty, color="blue")
ax.scatter(x, y, color="blue")
ax.plot(tx, predicted_y, color="red")
plt.show()
