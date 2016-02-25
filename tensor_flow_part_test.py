"""Tensorflow side of q learner"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform

import tensorflow as tf
import numpy as np
from losses import *


class InputTest(tf.test.TestCase):


    def testQLoss(self):
      action_inds_py = [[0, 0, 1], [0, 1, 0], [0, 0, 1]]
      y_values_py = [[0, 0, 1.0], [0, 2.0, 0], [0, 0, 0.0]]
      q_values_py = [[1230, 70, 1.1], [-2.0, -222, -100], [.2, -.2, 55.5]]
      

      with self.test_session():
          # Initialize variables for numpy implementation.
          action_inds_np = np.array(action_inds_py, dtype=np.float32)
          y_values_np = np.array(y_values_py, dtype=np.float32)
          q_values_np = np.array(q_values_py, dtype=np.float32)
          np_loss = q_learning_loss_numpy(q_values_np, y_values_np, action_inds_np)

          action_inds = tf.Variable(action_inds_np)
          y_values = tf.Variable(y_values_np)
          q_values = tf.Variable(q_values_np)
          tf_loss = q_learning_loss(q_values, y_values, action_inds, "q_loss")
          
          tf.initialize_all_variables().run()

          tf_loss_np = tf_loss.eval()
          print(tf_loss_np, np_loss)
          
          self.assertAllEqual(tf_loss_np, np_loss)
        
      


if __name__ == "__main__":
    tf.test.main()
