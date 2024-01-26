import tensorflow as tf


def amp_to_db(x, eps=1e-16, top_db=40):
    """
    Convert the input tensor from amplitude to decibel scale.

    Arguments:
        x {tf.Tensor} -- Input tensor.

    Keyword Arguments:
        eps {tf.Tensor} -- Small value to avoid numerical instability. (default: tf.constant(np.finfo(np.float64).eps))
        top_db {float} -- Threshold the output at "top_db" below the peak (default: 40)

    Returns:
        tf.Tensor -- Output tensor in decibel scale.
    """
    x_db = 20 * tf.math.log(tf.abs(x) + eps) / tf.math.log(10.0)

    return tf.maximum(x_db, (tf.reduce_max(x_db, axis=0) - top_db)[tf.newaxis, :])


def temperature_sigmoid(x, x0, temp_coeff):
    """
    Apply a sigmoid function with temperature scaling.

    Arguments:
        x {tf.Tensor} -- Input tensor.
        x0 {float} -- Parameter that controls the threshold of the sigmoid.
        temp_coeff {float} -- Parameter that controls the slope of the sigmoid.

    Returns:
        tf.Tensor -- Output tensor after applying the sigmoid with temperature scaling.
    """
    return tf.math.sigmoid((x - x0) / temp_coeff)


def linspace(start, stop, num=50, endpoint=True):
    """
    Generate a linearly spaced 1-D tensor.

    Arguments:
        start {Number} -- The starting value of the sequence.
        stop {Number} -- The end value of the sequence, unless `endpoint` is set to False.
                        In that case, the sequence consists of all but the last of `num + 1`
                        evenly spaced samples, so that `stop` is excluded. Note that the step
                        size changes when `endpoint` is False.

    Keyword Arguments:
        num {int} -- Number of samples to generate. Default is 50. Must be non-negative.
        endpoint {bool} -- If True, `stop` is the last sample. Otherwise, it is not included.
                          Default is True.

    Returns:
        tf.Tensor -- 1-D tensor of `num` equally spaced samples from `start` to `stop`.
    """
    if endpoint:
        return tf.linspace(start, stop, num)
    else:
        step = (stop - start) / num
        return tf.range(start, stop, step)
