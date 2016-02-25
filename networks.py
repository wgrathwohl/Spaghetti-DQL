"""
Houses networks
"""
from layers import *

def q_learning_model1(images, input_channels, num_outputs=4):
    """
    Model following the deepmind paper
    Args:
        images: Images returned from distorted_inputs() or inputs().
        is_training: True if training, false if eval
    Returns:
        Logits.
    """
    # conv1
    n_filters_conv1 = 16
    conv1 = conv_layer(images, "conv1", input_channels, n_filters_conv1, [5, 5], "MSFT", 0.004)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    n_filters_conv2 = 32
    conv2 = conv_layer(pool1, "conv2", n_filters_conv1, n_filters_conv2, [5, 5], "MSFT", 0.004)

    # pool2
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME', name='pool2'
    )

    # reshape pool2 to linear
    reshape, dim = reshape_conv_layer(pool2)

    # local3
    n_outputs_local_3 = 256
    local3 = linear_layer(reshape, "local3", dim, n_outputs_local_3, .01, 0.004)

    # outputs no nonlinearity on end
    outputs = linear_layer(local3, "q_values", n_outputs_local_3, num_outputs, .01, .004, use_nonlinearity=False)  

    return outputs