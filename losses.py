"""
Will house nessesary tensorflow parts here
"""
import tensorflow as tf
import numpy as np


def scale_labels(labels, margin=1):
    """
    Converts 0,1 labels to -margin,margin labels
    """
    return (2.0 * margin * labels) - margin


def hinge_loss(logits, labels, name=None):
    """
    Implements squared hinge loss
    """
    scaled_labels = scale_labels(labels)
    logits_labels = tf.mul(logits, scaled_labels)
    logits_labels_shifted = tf.minimum(logits_labels - 1.0, 0.0)
    squared_component_hinge_loss = tf.square(logits_labels_shifted)
    loss = tf.reduce_sum(squared_component_hinge_loss, 1)
    return loss

def q_learning_loss(q_values, target_values, target_inds, scope_name):
	"""
	q_values is q_values output [batch_size, n_outputs]
	target_value is the target [batch_size, n_outputs], should be all zeros
		except for 1 value y_a where a is the action chosen, then
		this returns (q_values[:, a] - y_a)^2
	target_inds is the one hot vector with the index where target_values
		is nonzero
	"""
	with tf.variable_scope(scope_name) as scope:
		# zero out all q_values for the actions we don't care about
		q_values_for_actions = tf.mul(q_values, target_inds)
		diff = q_values_for_actions - target_values
		loss_vector = tf.reduce_sum(diff, 1)
		l2 = tf.square(loss_vector)
		total_loss = tf.reduce_mean(l2, name=scope.name)
		return total_loss

def q_learning_loss_numpy(q_values, target_values, target_inds):
	q_values_for_actions = np.multiply(q_values, target_inds)
	diff = np.sum(q_values_for_actions - target_values, axis=1)
	l2 = diff**2
	loss = np.mean(l2)
	return loss







