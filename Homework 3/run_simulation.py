###
#Run simulation
###
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process
import tensorflow.python.saved_model
from tensorflow.contrib import predictor
from tensorflow.python.saved_model import tag_constants



env = gym.make('CartPole-v0')

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('~/my_test_model.meta')
	saver.restore(sess,tf.train.latest_checkpoint('~/'))
	for i in range(10):
		ob = env.reset()

#	for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#		print(i.name)


		done = False 
		while not done:
			env.render()
			ac = sess.run('sample:0', feed_dict={'ob:0': np.array([ob])})
			ob, rew, done, _ = env.step(ac)
