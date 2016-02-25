"""
houses neural network layers
"""
import tensorflow as tf
from utils import _variable_with_weight_decay, _variable_on_cpu, _activation_summary
from tensorflow.python.ops import control_flow_ops



def conv_layer(state_below, scope_name, n_inputs, n_outputs, filter_shape, stddev, wd):
    """
    A Standard convolutional layer
    """
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay(
            "weights", shape=[filter_shape[0], filter_shape[1], n_inputs, n_outputs],
            wd=wd
        )
        conv = tf.nn.conv2d(state_below, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu("biases", [n_outputs], tf.constant_initializer(0.0))
        bias = tf.add(conv, biases)
        output = tf.nn.relu(bias, name=scope.name)
        _activation_summary(output)
    return output

def reshape_conv_layer(state_below):
    """
    Reshapes a conv layer activations to be linear. Assumes that batch dimension is 0
    assumes that all other dimensions as held constant
    """
    dims = state_below.get_shape().as_list()
    # get back size as tensor so batch size can be dynamic!
    batch_size = tf.shape(state_below)[0]
    conv_dims = dims[1:]
    dim = 1
    for d in conv_dims:
        dim *= d
    reshape = tf.reshape(state_below, tf.pack([batch_size, dim]))
    return reshape, dim

def linear_layer(state_below, scope_name, n_inputs, n_outputs, stddev, wd, use_nonlinearity=True):
    """
    Standard linear neural network layer
    """
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay(
            'weights', [n_inputs, n_outputs],
            stddev=stddev, wd=wd
        )
        biases = _variable_on_cpu(
            'biases', [n_outputs], tf.constant_initializer(0.0)
        )
        activation = tf.nn.xw_plus_b(state_below, weights, biases, name="activation")
        if use_nonlinearity:
            output = tf.nn.relu(activation, name=scope.name)
        else:
            output = activation
        _activation_summary(output)
    return output
