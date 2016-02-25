"""
Will house classes that allow interaction with tensor flow parts of q-learning model
"""

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from layers import *
from utils import _add_loss_summaries
import losses

def preprocess_images(images_placeholder): # , processed_size):
    """
    Preprocess images for nn input
    Args:
        images_placeholder

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1] size.
        labels: Labels. 2D tensor of [batch_size, 6] size.
    """
    # cast to float
    float_image = tf.cast(images_placeholder, tf.float32)

    # height = processed_size[0]
    # width = processed_size[1]

    # # Image processing for evaluation.
    # # Crop the central [height, width] of the image.
    # # TODO : FIND OUT HOW TO DO THIS WITH WARPING IF NEED BE
    # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
    #                                                        height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(float_image)

    return float_image

class LiveQModel:
    def __init__(
        self, 
        q_values, 
        im_shape, num_channels, 
        batch_size,
        actions, train_dir,
        optimizer,
        decay_steps, initial_learning_rate, learning_rate_decay_factor=.1,
        checkpoint_path=None):
        """
        checkpoint_path: path to model checkpoint file
        logits is a function taking a tensor -> float array
        logits must take a single tensor argument
        ind_to_label is object such that ind_to_label[ind] = label
        """
        self.im_shape = im_shape
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.Session()                

            # variables to represent images, q values, expected q_values, and action inds
            self.images_placeholder = tf.placeholder("float", shape=[None, im_shape[0], im_shape[1], num_channels])
            self.images = self.images_placeholder #preprocess_images(self.images_placeholder)

            # TODO APPLY THIS NORMALIZATION ACROSS THE BATCH
            self.q_values = q_values(self.images, num_channels, len(actions))
            # TODO FIGURE OUT HOW TO REUSE THE VARIABLES BETWEEN THESE TWO OUTPUTS

            self.target_values = tf.placeholder("float", shape=[None, len(actions)])
            self.target_inds = tf.placeholder("float", shape=[None, len(actions)])

            # generate the loss and training up
            self.q_learning_loss = losses.q_learning_loss(self.q_values, self.target_values, self.target_inds, "q_loss")
            tf.add_to_collection('losses', self.q_learning_loss)
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            loss_averages_op = _add_loss_summaries(self.loss)

            self.global_step = tf.Variable(0, trainable=False)

            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(
                INITIAL_LEARNING_RATE,
                self.global_step,
                decay_steps,
                learning_rate_decay_factor,
                staircase=True
            )
            tf.scalar_summary('learning_rate', lr)

            # Compute gradients.
            with tf.control_dependencies([loss_averages_op]):
                opt = optimizer(lr) # tf.train.AdamOptimizer(lr)
                grads = opt.compute_gradients(self.loss)

            # Apply gradients.
            self.apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                tf.histogram_summary(var.op.name, var)

            # Add histograms for gradients.
            for grad, var in grads:
                if grad:
                    tf.histogram_summary(var.op.name + '/gradients', grad)

            with tf.control_dependencies([self.apply_gradient_op]):
                self.train_op = tf.no_op(name='train')


            # Build the summary operation based on the TF collection of Summaries.
            self.summary_op = tf.merge_all_summaries()
            
            self.saver = tf.train.Saver(tf.all_variables())
            self.actions = actions

            # Build an initialization operation to run below.
            init = tf.initialize_all_variables()

            # Start running operations on the Graph.
            self.sess.run(init)

            self.summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=self.sess.graph_def)

        if checkpoint_path is not None:
            # restore the model's parameters
            self.checkpoint_step = checkpoint_path.split('-')[-1]
            self.checkpoint_path = checkpoint_path
            self.saver.restore(self.sess, checkpoint_path)

    def training_iteration(self, images, target_values, target_inds):
        fd = {
            self.images_placeholder: images,
            self.target_values: target_values,
            self.target_inds: target_inds
        }
        start_time = time.time()
        _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict=fd)
        duration = time.time() - start_time
        print "Training iteration took: {} seconds".format(duration)
        print "Loss: {}".format(loss_value)
        return

    def get_q_values(self, image):
        """
        Runs the image through the network and returns the logits
        """
        assert len(image.shape) == 3
        im = np.expand_dims(image, axis=0)
        fd = {self.images_placeholder: im}
        q_values = self.sess.run(self.q_values, feed_dict=fd)
        return q_values[0]

    def get_q_values_multi(self, images):
        """
        Runs the images through the network and returns the logits
        """
        fd = {self.images_placeholder: images}
        q_values = self.sess.run(self.q_values, feed_dict=fd)
        return q_values

    def save_checkpoint(self, step):
        """
        Simply saves the model
        """
        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def write_summary(self, images, target_values, target_inds, step):
        """
        Writes a summary for tensorboard
        """
        fd = {
            self.images_placeholder: images,
            self.target_values: target_values,
            self.target_inds: target_inds
        }
        summary_str = self.sess.run(self.summary_op, feed_dict=fd)
        self.summary_writer.add_summary(summary_str, step)

